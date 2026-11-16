#!/bin/bash
# usage: waitjob.sh <job basename> [poll seconds]. Returns when the laptop queue has moved the job out of pending and running; prints its log tail.
job=$1; poll=${2:-90}; O=(-o BatchMode=yes -o ConnectTimeout=15)
for i in $(seq 1 1200); do
  st=$(ssh "${O[@]}" anweat-ip "if (Test-Path qwen-all\\queue\\done\\$job.ps1) {'done'} elseif (Test-Path qwen-all\\queue\\failed\\$job.ps1) {'failed'} else {'waiting'}" 2>/dev/null | tr -d '\r')
  case "$st" in done|failed) echo "JOB $job: $st"; ssh "${O[@]}" anweat-ip "Get-Content qwen-all\\queue\\logs\\$job.log -Tail 6 -Encoding Unicode" 2>/dev/null | cut -c1-160; exit 0;; esac
  sleep "$poll"
done
echo "TIMEOUT waiting for $job"
