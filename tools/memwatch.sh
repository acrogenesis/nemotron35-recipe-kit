#!/bin/bash
# Record host + GPU memory while a server starts and serves. Start before launch.
# Host memory uses decimal GB from /proc/meminfo KiB; GPU memory uses MiB.
OUT=${1:-memwatch.csv}
echo "t,mem_available_gb,swap_used_gb,gpu_used_mib" > "$OUT"
while true; do
  read -r ma st sf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
  g=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
    awk '/^[0-9]+([.][0-9]+)?$/{sum+=$1;n++}END{if(n)printf "%.0f",sum}')
  awk -v t="$(date +%s)" -v ma="$ma" -v st="$st" -v sf="$sf" -v gpu="$g" \
    'BEGIN{printf "%s,%.6f,%.6f,%s\n",t,ma*1024/1e9,(st-sf)*1024/1e9,gpu}' >> "$OUT"
  sleep 1
done
