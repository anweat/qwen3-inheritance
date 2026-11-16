# Sequential experiment queue for the laptop. Jobs are ps1 files in queue\pending, run in name order.
# GPU jobs wait while something else holds GPU memory; names containing nogpu skip that check.
# Number job names so the order is explicit, for example 01_nogpu_env.ps1, 02_structure.ps1.
# To stop after the current job, create queue\STOP. The runner exits on its own after 12 idle hours.
$ErrorActionPreference = 'Continue'
$root = "$HOME\qwen-all\queue"
foreach ($d in 'pending','running','done','failed','logs') { New-Item -ItemType Directory -Force "$root\$d" | Out-Null }
$idle = 0
"runner started $(Get-Date -Format s) pid $PID" | Set-Content "$root\runner.status"
while (-not (Test-Path "$root\STOP") -and $idle -lt 1440) {
    $job = Get-ChildItem "$root\pending\*.ps1" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 1
    if (-not $job) { $idle++; Start-Sleep 30; continue }
    $idle = 0
    if ($job.Name -notlike '*nogpu*') {
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($used -gt 2000) {
            "waiting: GPU memory $used MiB in use before $($job.Name), $(Get-Date -Format s)" | Set-Content "$root\runner.status"
            Start-Sleep 60
            continue
        }
    }
    $run = "$root\running\$($job.Name)"
    Move-Item $job.FullName $run -Force
    $log = "$root\logs\$($job.BaseName).log"
    "running $($job.Name) since $(Get-Date -Format s)" | Set-Content "$root\runner.status"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $run *> $log
    $code = $LASTEXITCODE
    "[runner] exit code $code at $(Get-Date -Format s)" | Out-File $log -Append -Encoding unicode
    $dest = if ($code -eq 0) { 'done' } else { 'failed' }
    Move-Item $run "$root\$dest\$($job.Name)" -Force
    "finished $($job.Name) -> $dest, $(Get-Date -Format s)" | Set-Content "$root\runner.status"
}
"runner exited $(Get-Date -Format s)" | Set-Content "$root\runner.status"
