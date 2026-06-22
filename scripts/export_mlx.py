#!/usr/bin/env python3
"""
Merge the Production LoRA adapter into granite-docling, convert to MLX, verify,
and push to HuggingFace. Designed to run on a macOS Apple-Silicon machine
(e.g. a Codemagic mac_mini_m2) where MLX runs natively on Metal.

Config via environment variables (set as a Codemagic env group):
  VM_IP              GCP VM public IP (MLflow :5000 / MinIO :9000)
  MINIO_ACCESS_KEY   MinIO access key
  MINIO_SECRET_KEY   MinIO secret key
  HF_TOKEN           HuggingFace write token
  HF_TARGET          target MLX repo (default below)
  OCR_TEST_IMAGE     image used for the verify step (default scripts/ttcp1.jpg)
  PUSH               '1' to push to HF (default), '0' to skip
"""
import os
import sys
import shutil

import mlflow
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import snapshot_download, HfApi
from safetensors.torch import load_file, save_file

BASE_MODEL = os.environ.get('BASE_MODEL', 'ibm-granite/granite-docling-258M')
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

# 2. Merge (bf16). granite-docling ties lm_head to the input embeddings, so it
#    is normally not stored separately and mlx_vlm.convert reports it missing.
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
emb = model.get_input_embeddings().weight
model.lm_head.weight = nn.Parameter(emb.detach().clone())
model.config.tie_word_embeddings = False
if hasattr(model.config, 'text_config'):
    model.config.text_config.tie_word_embeddings = False
# Stop transformers from treating lm_head as a tied (droppable) weight on save.
for m in [model, *model.modules()]:
    if hasattr(m, '_tied_weights_keys'):
        m._tied_weights_keys = []
model.save_pretrained(MERGED_DIR, safe_serialization=True)

# 2b. Bulletproof (version-independent): if lm_head still isn't in the
#     safetensors, inject it = a copy of the text input embeddings.
st = os.path.join(MERGED_DIR, 'model.safetensors')
if os.path.exists(st):
    sd = load_file(st)
    print('DEBUG head/embed keys:',
          [k for k in sd if 'lm_head' in k or k.endswith('embed_tokens.weight')],
          flush=True)
    emb_key = next(k for k in sd if k.endswith('embed_tokens.weight'))
    # Inject the POST-sanitize key directly. mlx_vlm's idefics3 expects the mlx
    # param 'language_model.lm_head.weight'; this exact key passes every sanitize
    # rule unchanged, so it matches regardless of the installed mlx_vlm version.
    # Drop the other head variants to avoid a duplicate after sanitize.
    for k in ('lm_head.weight', 'model.lm_head.weight',
              'language_model.lm_head.weight'):
        sd.pop(k, None)
    sd['language_model.lm_head.weight'] = sd[emb_key].clone()  # clone: no shared mem
    save_file(sd, st, metadata={'format': 'pt'})
    _chk = list(load_file(st).keys())
    print('OCRDEBUG file keys w/ lm_head:',
          [k for k in _chk if 'lm_head' in k], flush=True)
    assert 'language_model.lm_head.weight' in _chk, 'INJECT NOT PERSISTED to ' + st
else:
    print('WARN: sharded safetensors — lm_head fix skipped', flush=True)

# 3. Copy tokenizer + processor config from the base (AutoProcessor.save can
#    fail on bleeding-edge transformers; keep the merged config.json).
src = snapshot_download(
    BASE_MODEL,
    allow_patterns=['*.json', '*.txt', '*.model', 'tokenizer*',
                    'preprocessor_config.json', 'processor_config.json',
                    'chat_template*', 'merges.txt', 'vocab.json',
                    'special_tokens_map.json', 'added_tokens.json'],
)
for f in os.listdir(src):
    s = os.path.join(src, f)
    if os.path.isfile(s) and f != 'config.json':
        shutil.copy(s, os.path.join(MERGED_DIR, f))
print('merged:', sorted(os.listdir(MERGED_DIR)), flush=True)

# 4. Convert HF -> MLX (no quantization).
import mlx_vlm  # noqa: E402
print('OCRDEBUG mlx_vlm version:', mlx_vlm.__version__, flush=True)
assert run(f'{sys.executable} -m mlx_vlm convert '
           f'--hf-path {MERGED_DIR} --mlx-path {MLX_DIR}') == 0, 'convert failed'

# 5. VERIFY — generate on the test page; compare the header with the serving
#    output in the build log (ỦY BAN NHÂN DÂN / CỘNG HÒA… / Chủ tịch / signature).
print('\n================ VERIFY (compare with serving) ================',
      flush=True)
if os.path.exists(IMG):
    run(f'{sys.executable} -m mlx_vlm generate --model {MLX_DIR} --image {IMG} '
        f'--prompt "Convert this page to docling format." '
        f'--temperature 0.0 --max-tokens 4096')
else:
    print(f'(skip verify — {IMG} not found; commit a test image to enable)',
          flush=True)

# 6. Push the MLX folder to HuggingFace.
if os.environ.get('PUSH', '1') == '1':
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(HF_TARGET, repo_type='model', exist_ok=True)
    api.upload_folder(folder_path=MLX_DIR, repo_id=HF_TARGET)
    print('pushed ->', f'https://huggingface.co/{HF_TARGET}', flush=True)