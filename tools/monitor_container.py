"""Record cgroup memory/swap and host CPU use without entering the container."""
import argparse
import csv
import os
from pathlib import Path
import time

ap = argparse.ArgumentParser()
ap.add_argument('--pid', type=int, required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
group = next(line.split(':', 2)[2] for line in Path(f'/proc/{a.pid}/cgroup').read_text().splitlines()
             if line.startswith('0::'))
root = Path('/sys/fs/cgroup') / group.lstrip('/')


def cpu():
    values = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    return sum(values[:8]), values[3]+values[4]


def number(name):
    try:
        return int((root/name).read_text())
    except (FileNotFoundError, ValueError):
        return None


def container_cpu():
    return int(dict(line.split() for line in (root/'cpu.stat').read_text().splitlines())['usage_usec'])


previous = cpu()
previous_usage = container_cpu()
previous_time = time.monotonic()
with open(a.out, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'memory_current', 'memory_peak', 'swap_current',
        'anonymous_bytes', 'file_cache_bytes', 'host_cpu_percent', 'container_cpu_percent', 'other_host_cpu_percent', 'oom', 'oom_kill'])
    writer.writeheader()
    while root.exists():
        now = cpu(); total = now[0]-previous[0]; idle = now[1]-previous[1]
        usage, sample_time = container_cpu(), time.monotonic()
        elapsed = sample_time-previous_time
        container_percent = 100*(usage-previous_usage)/1e6/elapsed/os.cpu_count()
        host_percent = 100*(total-idle)/total if total else 0
        stat = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
        events = dict(line.split() for line in (root/'memory.events').read_text().splitlines())
        writer.writerow({'time': time.time(), 'memory_current': number('memory.current'),
            'memory_peak': number('memory.peak'), 'swap_current': number('memory.swap.current'),
            'anonymous_bytes': stat.get('anon'), 'file_cache_bytes': stat.get('file'),
            'host_cpu_percent': round(host_percent, 2),
            'container_cpu_percent': round(container_percent, 2),
            'other_host_cpu_percent': round(max(0, host_percent-container_percent), 2),
            'oom': events.get('oom'), 'oom_kill': events.get('oom_kill')})
        f.flush(); previous = now; previous_usage = usage; previous_time = sample_time; time.sleep(1)
