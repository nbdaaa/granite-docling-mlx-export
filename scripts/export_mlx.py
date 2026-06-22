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
  4. ship it alongside ALL of official's config/processor/tokenizer files.

The result loads the same way the official model does in mlx_vlm AND mlx-swift,
so both prior blockers (lm_head not binding, AutoProcessor "Unrecognized image
processor") disappear.

Config via environment variables (Codemagic env group):
  VM_IP, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, HF_TOKEN
  HF_TARGET   target MLX repo (default below)
  OCR_TEST_IMAGE  verify image (default scripts/ttcp1.jpg)
  PUSH        '1' to push to HF (default), '0' to skip
"""
import os
import re
import sys
import shutil

import mlflow
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import snapshot_download, HfApi

BASE_MODEL = os.environ.get('BASE_MODEL', 'ibm-granite/granite-docling-258M')
OFFICIAL_MLX = os.environ.get('OFFICIAL_MLX', 'ibm-granite/granite-docling-258M-mlx')
MODEL_NAME = os.environ.get('MODEL_NAME', 'granite-docling-adapter')
HF_TARGET = os.environ.get('HF_TARGET', 'nbdaaa/granite-docling-258M-mine-mlx')
MERGED_DIR = 'granite-docling-merged'
MLX_DIR = 'granite-docling-mlx'
IMG = os.environ.get('OCR_TEST_IMAGE', 'scripts/ttcp1.jpg')

VM = os.environ['VM_IP']
os.environ['MLFLOW_TRACKING_URI'] = f'http://{VM}:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{VM}:9000'
os.environ['AWS_ACCESS_KEY_ID'] = os.environ['MINIO_ACCESS_KEY']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['MINIO_SECRET_KEY']
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


# 1. Pull the Production adapter from MLflow.
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
client = mlflow.MlflowClient()
run_id = next(
    (v.run_id for v in client.search_model_versions(f"name='{MODEL_NAME}'")
     if v.current_stage == 'Production'),
    None,
)
assert run_id, f'No Production version for {MODEL_NAME}'
ADAPTER_DIR = client.download_artifacts(run_id, 'adapter', 'adapter_dl')
print('adapter:', ADAPTER_DIR, os.listdir(ADAPTER_DIR), flush=True)

# 2. Download the official MLX repo — our structural template (config, processor,
#    tokenizer, and the canonical 471-key layout that mlx_vlm/mlx-swift load).
OFFICIAL_DIR = snapshot_download(OFFICIAL_MLX)
print('official mlx files:', sorted(os.listdir(OFFICIAL_DIR)), flush=True)

# 3. Merge our adapter (bf16). Untie + materialize lm_head so it lands in the
#    state dict; granite ties lm_head to the input embeddings.
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
emb = model.get_input_embeddings().weight
model.lm_head.weight = nn.Parameter(emb.detach().clone())
model.config.tie_word_embeddings = False
if hasattr(model.config, 'text_config'):
    model.config.text_config.tie_word_embeddings = False
for m in [model, *model.modules()]:
    if hasattr(m, '_tied_weights_keys'):
        m._tied_weights_keys = []
model.save_pretrained(MERGED_DIR, safe_serialization=True)

# 4. Load official + merged weights with MLX; sanitize our keys to official names.
import mlx.core as mx  # noqa: E402

official_st = os.path.join(OFFICIAL_DIR, 'model.safetensors')
official = mx.load(official_st)
official_keys = set(official.keys())
print('official keys:', len(official_keys), flush=True)

merged_st = os.path.join(MERGED_DIR, 'model.safetensors')
assert os.path.exists(merged_st), 'merged is sharded — expected single safetensors'
raw = mx.load(merged_st)
ours = {sanitize_key(k): v for k, v in raw.items()}
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
    if v.shape != official[k].shape:
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

# 7. Push the MLX folder to HuggingFace.
if os.environ.get('PUSH', '1') == '1':
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(HF_TARGET, repo_type='model', exist_ok=True)
    api.upload_folder(folder_path=MLX_DIR, repo_id=HF_TARGET)
    print('pushed ->', f'https://huggingface.co/{HF_TARGET}', flush=True)