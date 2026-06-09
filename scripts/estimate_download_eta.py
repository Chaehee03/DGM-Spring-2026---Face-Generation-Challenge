import os
import time
import subprocess

DIR = "/data/chsong/dgm-face/celebvhq_hf"

# 예상 전체 크기. 필요하면 수정.
TOTAL_GB = 40.0
INTERVAL = 60

def get_size_bytes(path):
    out = subprocess.check_output(["du", "-sb", path]).decode().split()[0]
    return int(out)

s1 = get_size_bytes(DIR)
time.sleep(INTERVAL)
s2 = get_size_bytes(DIR)

downloaded_gb = s2 / 1024**3
delta_mb = (s2 - s1) / 1024**2
speed_mb_s = delta_mb / INTERVAL

remaining_gb = max(TOTAL_GB - downloaded_gb, 0)
remaining_mb = remaining_gb * 1024

if speed_mb_s > 0:
    eta_sec = remaining_mb / speed_mb_s
    eta_hr = eta_sec / 3600
else:
    eta_hr = float("inf")

print(f"Current downloaded: {downloaded_gb:.2f} GB")
print(f"Assumed total: {TOTAL_GB:.2f} GB")
print(f"Speed: {speed_mb_s:.2f} MB/s")
print(f"Remaining: {remaining_gb:.2f} GB")
print(f"ETA: {eta_hr:.2f} hours")
