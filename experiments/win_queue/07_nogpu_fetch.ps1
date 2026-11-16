$ErrorActionPreference='Continue'
Set-Location $HOME\qwen-all
$V="$HOME\qwen-all\.venv-tts\Scripts\python.exe"
if (-not (Test-Path $V)) { 'venv missing'; exit 9 }
$env:PYTHONUTF8='1'
& $V -u queue\fetch_direct.py 2>&1 | ForEach-Object { "$_" }
exit $LASTEXITCODE
