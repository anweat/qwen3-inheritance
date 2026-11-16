$ErrorActionPreference='Continue'
Set-Location $HOME\qwen-all
$PY="$HOME\anaconda3\python.exe"; $env:PYTHONUTF8='1'; $env:UV_HTTP_TIMEOUT='600'
$V="$HOME\qwen-all\.venv-tts\Scripts\python.exe"
if (-not (Test-Path $V)) { & $PY -m uv venv --python 3.12 .venv-tts 2>&1 | ForEach-Object { "$_" } }
& $PY -m uv pip install --python $V torch torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | Select-Object -Last 2 | ForEach-Object { "$_" }
& $PY -m uv pip install --python $V --index-url https://pypi.tuna.tsinghua.edu.cn/simple qwen-tts scipy safetensors huggingface_hub 2>&1 | Select-Object -Last 3 | ForEach-Object { "$_" }
& $V -c "import torch,transformers,qwen_tts;print('torch',torch.__version__,torch.cuda.is_available(),'tf',transformers.__version__,'qwen_tts ok')" 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { exit 1 }
$env:HF_ENDPOINT='https://hf-mirror.com'; $env:HF_HUB_DISABLE_XET='1'
& $V -u queue\fetch_tts.py 2>&1 | Where-Object { "$_" -notmatch 'it/s\]|Fetching|Redirects' } | ForEach-Object { "$_" }
exit $LASTEXITCODE
