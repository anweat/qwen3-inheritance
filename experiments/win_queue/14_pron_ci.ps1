$ErrorActionPreference='Continue'
Set-Location $HOME\qwen-all\e4_tts_structure
$V="$HOME\qwen-all\.venv-tts\Scripts\python.exe"
if (-not (Test-Path $V)) { 'venv missing'; exit 9 }
$env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'; $env:HF_HUB_OFFLINE='1'
& $V -u pron_ci.py 2>&1 | ForEach-Object { "$_" }
exit $LASTEXITCODE
