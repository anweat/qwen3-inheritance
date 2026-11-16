#!/usr/bin/env bash
# E7 on an AutoDL Pro instance: probe download routes, build the env, fetch models, run stages 1-2.
# Layout: /root/e7/{e7_cross_family,e2_activation}. Logs in /root/e7/logs; writes DONE or FAILED there.
set -u
export PATH=/root/miniconda3/bin:$PATH  # non-login ssh shells do not see conda's python
R=/root/e7; L=$R/logs; mkdir -p $L; cd $R/e7_cross_family
step() { echo "== $(date +%T) $*" | tee -a $L/steps.log; }
fail() { step "FAILED: $*"; touch $L/FAILED; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $L/gpu.txt 2>&1
df -h / > $L/disk.txt

step probe
if [ ! -s $L/probe.txt ]; then
probe() { curl -sL -r 0-31457279 -o /dev/null -m 40 -w "%{http_code} %{size_download} %{time_total}" "$1" | awk -v u="$1" -v t="$2" '{printf "%-6s %-90s code %s %.1f MB/s\n", t, u, $1, $2/1e6/($3+1e-9)}'; }
U1=https://modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/master/model.safetensors
U2=https://hf-mirror.com/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors
U3=https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors
for u in $U1 $U2 $U3; do probe $u plain; done > $L/probe.txt
[ -f /etc/network_turbo ] && (source /etc/network_turbo >/dev/null; for u in $U1 $U2 $U3; do probe $u turbo; done) >> $L/probe.txt
fi
cat $L/probe.txt | tee -a $L/steps.log

step env
PY=$R/.venv/bin/python
# 2026-11-11: uv against the image's http aliyun index stalled (300 MB in 40 min); mirrors themselves give ~7 MB/s
# on a 40 MB range read (ustc 7.5, aliyun 6.9, tuna 6.2, tencent 1.4). Use ustc over https with few parallel downloads.
if ! $PY -c "import torch, transformers" 2>/dev/null; then
  python3 -m pip install -q uv >> $L/env.log 2>&1 || fail "pip install uv"
  [ -x $PY ] || ( [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null; python3 -m uv venv $R/.venv -p 3.11 ) >> $L/env.log 2>&1 || fail "python 3.11"
  UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_TIMEOUT=600 python3 -m uv pip install --python $PY --index-url https://mirrors.ustc.edu.cn/pypi/simple \
    torch==2.5.1 transformers==4.57.1 accelerate safetensors scipy numpy huggingface_hub >> $L/env.log 2>&1 || fail "pip deps"
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tee -a $L/steps.log

step fetch
while pgrep -f 'python3 fetch_models.py' >/dev/null; do sleep 20; done  # a parallel pre-fetch may still be running
# probe on 2026-11-11: ModelScope 4.6 MB/s plain, 13.6 MB/s with academic acceleration
( [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null; python3 fetch_models.py $R/models ) > $L/fetch.log 2>&1 || fail fetch
grep -q FAILED $L/fetch.log && fail "fetch has failed files"
cat $R/models/manifest.txt >> $L/steps.log

step weights
HF_HUB_OFFLINE=1 $PY weights.py $R/models > $L/weights2.txt 2>&1 || fail weights
step cka
HF_HUB_OFFLINE=1 $PY cka.py $R/models > $L/cka2.txt 2>&1 || fail cka
step done
touch $L/DONE
