"""Poll a Kaggle kernel until it finishes. Exit 1 if it errors or times out."""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("slug")
ap.add_argument("--minutes", type=int, default=340)
a = ap.parse_args()
end = time.time() + a.minutes * 60
time.sleep(60)
while time.time() < end:
    r = subprocess.run(["kaggle", "kernels", "status", a.slug], capture_output=True, text=True)
    line = (r.stdout + r.stderr).strip()
    print(time.strftime("%H:%M:%S"), line, flush=True)
    low = line.lower()
    if "complete" in low:
        sys.exit(0)
    if "error" in low or "cancel" in low:
        sys.exit(1)
    time.sleep(120)
print("timed out waiting; the kernel may still be running on Kaggle")
sys.exit(1)
