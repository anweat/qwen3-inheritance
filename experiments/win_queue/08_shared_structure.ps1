$ErrorActionPreference='Continue'
Set-Location $HOME\qwen-all\e4_tts_structure
$V="$HOME\qwen-all\.venv-tts\Scripts\python.exe"
if (-not (Test-Path $V)) { 'venv missing'; exit 9 }
$env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'; $env:HF_HUB_OFFLINE='1'
& $V -u shared_structure.py 2>&1 | Where-Object { "$_" -notmatch 'Redirects|NativeCommandError|CategoryInfo|所在位置|FullyQualified|^\s*\+|it/s\]' } | ForEach-Object { "$_" }
exit $LASTEXITCODE
