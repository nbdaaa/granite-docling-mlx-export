#!/usr/bin/env python3
"""
Merge the Production LoRA adapter into granite-docling and produce an MLX model
that is structurally identical to IBM's official MLX release
(ibm-granite/granite-docling-258M-mlx) — only the tensor VALUES are our
fine-tuned weights. We do NOT use `mlx_vlm convert` / `load_weights` (which
mis-handles this model's head). Instead we:

  1. download the official MLX repo (known-good config + processor + 471 keys),
  2. merge our adapter (bf16) and sanitize the HF keys to the official MLX names,
  3. write a flat safetensors with EXACTLY the official key set, our values,
  4. ship it alongside ALL of official's config/processor/tokenizer files,
  5. patch the shipped config to route mlx-swift through the SmolVLM *tiling*
     processor (model_type=smolvlm) so the on-device app tiles like serving.

The result loads the same way the official model does in mlx_vlm AND mlx-swift,
so both prior blockers (lm_head not binding, AutoProcessor "Unrecognized image
processor") disappear.

Config via environment variables (Codemagic env group):
  HF_TOKEN    HuggingFace token (read for the adapter repo, write for the target)
  ADAPTER_REPO  source HF repo of the LoRA adapter (default below)
  ADAPTER_REV   adapter repo commit/revision to pin (default below)
  HF_TARGET   target MLX repo (default below)
  OCR_TEST_IMAGE  verify image (default scripts/ttcp1.jpg)
  PUSH        '1' to push to HF (default), '0' to skip
"""
import os
import re
import sys
import shutil

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import snapshot_download, HfApi

BASE_MODEL = os.environ.get('BASE_MODEL', 'ibm-granite/granite-docling-258M')
OFFICIAL_MLX = os.environ.get('OFFICIAL_MLX', 'ibm-granite/granite-docling-258M-mlx')
ADAPTER_REPO = os.environ.get('ADAPTER_REPO', 'nbdaaa/docling-ocr')
ADAPTER_REV = os.environ.get(
    'ADAPTER_REV', '68e4d5c80258a4c2ddf07da65af4da453ae66690')
HF_TARGET = os.environ.get('HF_TARGET', 'nbdaaa/granite-docling-258M-mine-mlx')
MERGED_DIR = 'granite-docling-merged'
MLX_DIR = 'granite-docling-mlx'
IMG = os.environ.get('OCR_TEST_IMAGE', 'scripts/ttcp1.jpg')

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'


def run(cmd: str) -> int:
    print(f'\n$ {cmd}', flush=True)
    return os.system(cmd)


def sanitize_key(k: str) -> str:
    """Replicate mlx_vlm idefics3 Model.sanitize key remapping exactly."""
    if re.match(r'^model\.', k):
        k = k.split('.', 1)[1]
    elif re.match(r'^lm_head\.', k):
        k = 'language_model.' + k
    if re.match(r'^text_model\.', k):
        k = 'language_model.' + k.split('.', 1)[1]
    return k


# 1. Download the LoRA adapter straight from the HF repo at a pinned commit.
ADAPTER_DIR = snapshot_download(
    ADAPTER_REPO, revision=ADAPTER_REV, token=os.environ.get('HF_TOKEN'))
# Point ADAPTER_DIR at the folder that actually holds adapter_config.json
# (in case the adapter lives in a subdirectory of the repo).
if not os.path.exists(os.path.join(ADAPTER_DIR, 'adapter_config.json')):
    for _root, _dirs, _files in os.walk(ADAPTER_DIR):
        if 'adapter_config.json' in _files:
            ADAPTER_DIR = _root
            break
print('adapter:', ADAPTER_REPO, ADAPTER_REV, '->', ADAPTER_DIR,
      os.listdir(ADAPTER_DIR), flush=True)

# 2. Download the official MLX repo — our structural template (config, processor,
#    tokenizer, and the canonical 471-key layout that mlx_vlm/mlx-swift load).
OFFICIAL_DIR = snapshot_download(OFFICIAL_MLX)
print('official mlx files:', sorted(os.listdir(OFFICIAL_DIR)), flush=True)

# 3. Merge our adapter (bf16). Untie lm_head (granite ties it to the input
#    embeddings) so it shows up as its own entry in the state dict.
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
emb = model.get_input_embeddings().weight
try:
    model.lm_head.weight = nn.Parameter(emb.detach().clone())
except Exception as e:
    print('lm_head untie skipped:', e, flush=True)
model.eval()

# 4. Convert the merged torch state_dict -> MLX arrays IN-PROCESS. We avoid
#    save_pretrained (its safetensors writer trips on this model's shared
#    tensors). numpy has no bf16, so route through fp32; we cast back to the
#    official bf16 dtype per-key below (bf16 -> fp32 -> bf16 is lossless).
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

official_st = os.path.join(OFFICIAL_DIR, 'model.safetensors')
official = mx.load(official_st)
official_keys = set(official.keys())
print('official keys:', len(official_keys), flush=True)

sd = model.state_dict()
ours = {}
for k, v in sd.items():
    ours[sanitize_key(k)] = mx.array(v.detach().to(torch.float32).cpu().numpy())
del sd, model, base
if 'language_model.lm_head.weight' not in ours:  # tied fallback
    ours['language_model.lm_head.weight'] = ours['language_model.embed_tokens.weight']

# 5. Build output with EXACTLY the official key set + our values + official dtype.
missing = sorted(official_keys - set(ours.keys()))
extra = sorted(set(ours.keys()) - official_keys)
print('MISSING from ours (%d): %s' % (len(missing), missing[:20]), flush=True)
print('EXTRA in ours  (%d): %s' % (len(extra), extra[:20]), flush=True)
assert not missing, 'our weights do not cover every official key'

out = {}
for k in official_keys:
    v = ours[k]
    if tuple(v.shape) != tuple(official[k].shape):
        # MLX stores conv weights channels-last (O,H,W,I); torch is (O,I,H,W).
        if v.ndim == 4 and tuple(mx.transpose(v, (0, 2, 3, 1)).shape) == tuple(official[k].shape):
            v = mx.transpose(v, (0, 2, 3, 1))
        else:
            raise SystemExit(f'shape mismatch {k}: ours {v.shape} vs official {official[k].shape}')
    out[k] = v.astype(official[k].dtype)

os.makedirs(MLX_DIR, exist_ok=True)
mx.save_safetensors(os.path.join(MLX_DIR, 'model.safetensors'), out)

# Ship ALL official non-weight files (config, processor, tokenizer, index, …).
for f in os.listdir(OFFICIAL_DIR):
    s = os.path.join(OFFICIAL_DIR, f)
    if os.path.isfile(s) and f not in ('model.safetensors', '.gitattributes', 'README.md'):
        shutil.copy(s, os.path.join(MLX_DIR, f))
print('MLX saved ->', MLX_DIR, sorted(os.listdir(MLX_DIR)), flush=True)

# 6. VERIFY — generate on the test page; compare header with serving output.
print('\n================ VERIFY (compare with serving) ================', flush=True)
if os.path.exists(IMG):
    run(f'{sys.executable} -m mlx_vlm generate --model {MLX_DIR} --image {IMG} '
        f'--prompt "Convert this page to docling format." '
        f'--temperature 0.0 --max-tokens 4096')
else:
    print(f'(skip verify — {IMG} not found)', flush=True)

# 6b. Patch the shipped config so mlx-swift routes through the SmolVLM *tiling*
#     processor. In mlx-swift-lm, SmolVLM2 == Idefics3 (same model/config); only
#     the processor differs — and granite's Idefics3Processor there has no tiling
#     and a broken image-token merge (crashes). model_type=smolvlm picks
#     SmolVLMProcessor (real tiling, correct token handling). Also cap
#     size.longest_edge to bound on-device prefill memory. The model VALUES /
#     471-key layout are untouched, so this only changes routing metadata.
import json as _json2  # noqa: E402
_cfgp = os.path.join(MLX_DIR, 'config.json')
_cfg = _json2.load(open(_cfgp))
_cfg['model_type'] = 'smolvlm'
_json2.dump(_cfg, open(_cfgp, 'w'), indent=2)

_ppp = os.path.join(MLX_DIR, 'preprocessor_config.json')
_pp = _json2.load(open(_ppp))
_pp['processor_class'] = 'SmolVLMProcessor'
_pp['video_sampling'] = {'fps': 1, 'max_frames': 64}  # required by SmolVLMProcessorConfiguration
_pp['image_seq_len'] = 64
if isinstance(_pp.get('size'), dict):
    _pp['size']['longest_edge'] = 1536
_json2.dump(_pp, open(_ppp, 'w'), indent=2)
print('patched config -> smolvlm / SmolVLMProcessor / size', _pp.get('size'), flush=True)

# 7. Push the MLX folder to HuggingFace.
if os.environ.get('PUSH', '1') == '1':
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(HF_TARGET, repo_type='model', exist_ok=True)
    api.upload_folder(folder_path=MLX_DIR, repo_id=HF_TARGET)
    print('pushed ->', f'https://huggingface.co/{HF_TARGET}', flush=True)