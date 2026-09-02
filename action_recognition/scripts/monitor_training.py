"""
monitor_training.py
===================
Real-time STGCN training progress monitor.
Reads the mmengine log file and displays a live dashboard every N seconds.

Run in a separate terminal:
    python monitor_training.py
"""

import os
import re
import time
import glob

LOG_DIR      = r"C:\Users\aryan\Downloads\BTP\mmaction2\work_dirs\stgcn_kabaddi"
TOTAL_EPOCHS = 80
ITERS_EPOCH  = 98     # 39 samples x10 repeat / batch_size 4
REFRESH_SEC  = 5      # update every 5 seconds

# Regex patterns
RE_TRAIN = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\]"
    r".*?lr:\s*([\d.e+-]+)"
    r".*?loss:\s*([\d.]+)"
    r".*?top1_acc:\s*([\d.]+)"
)
RE_VAL = re.compile(
    r"Epoch\(val\)\s+\[(\d+)\].*?acc/top1:\s*([\d.]+)"
)
RE_BEST = re.compile(
    r"The best checkpoint with ([\d.]+) acc/top1 at (\d+) epoch"
)


def find_log():
    """Find the most recent .log file under LOG_DIR."""
    pattern = os.path.join(LOG_DIR, "**", "*.log")
    logs = glob.glob(pattern, recursive=True)
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def bar(fraction, width=40):
    """Render a progress bar."""
    filled = int(fraction * width)
    empty  = width - filled
    pct    = fraction * 100
    return f"[{'#' * filled}{'.' * empty}] {pct:5.1f}%"


def parse_log(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_train = None
    val_results = {}   # epoch -> top1
    best_top1   = 0.0
    best_epoch  = 0

    for line in lines:
        m = RE_TRAIN.search(line)
        if m:
            last_train = {
                "epoch": int(m.group(1)),
                "iter":  int(m.group(2)),
                "total_iter": int(m.group(3)),
                "lr":    float(m.group(4)),
                "loss":  float(m.group(5)),
                "top1":  float(m.group(6)),
            }

        m = RE_VAL.search(line)
        if m:
            ep   = int(m.group(1))
            top1 = float(m.group(2))
            val_results[ep] = top1

        m = RE_BEST.search(line)
        if m:
            best_top1  = float(m.group(1))
            best_epoch = int(m.group(2))

    return last_train, val_results, best_top1, best_epoch


def display(log_path):
    last_train, val_results, best_top1, best_epoch = parse_log(log_path)

    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 65)
    print("   ST-GCN KABADDI  |  12 Classes  |  80 Epochs  |  RTX 2050")
    print("=" * 65)

    if last_train is None:
        print("  Waiting for training to start...")
        return

    ep   = last_train["epoch"]
    it   = last_train["iter"]
    tot  = last_train["total_iter"]
    lr   = last_train["lr"]
    loss = last_train["loss"]
    top1 = last_train["top1"]

    # ── Overall progress ──────────────────────────────────────────────────────
    overall_done  = (ep - 1) * ITERS_EPOCH + it
    overall_total = TOTAL_EPOCHS * ITERS_EPOCH
    overall_frac  = min(overall_done / overall_total, 1.0)

    print(f"\n  OVERALL PROGRESS")
    print(f"  {bar(overall_frac)}")
    print(f"  Epochs done : {ep-1} / {TOTAL_EPOCHS}   "
          f"({(ep-1)/TOTAL_EPOCHS*100:.1f}%)")
    print(f"  Epochs left : {TOTAL_EPOCHS - ep + 1}")

    # ── Current epoch ─────────────────────────────────────────────────────────
    ep_frac = it / tot
    print(f"\n  CURRENT EPOCH  [{ep} / {TOTAL_EPOCHS}]")
    print(f"  {bar(ep_frac)}")
    print(f"  Iterations  : {it} / {tot}   ({ep_frac*100:.1f}%)")
    print(f"  Iters left  : {tot - it}")

    # ── Live metrics ──────────────────────────────────────────────────────────
    print(f"\n  LIVE METRICS (epoch {ep})")
    print(f"  Loss        : {loss:.4f}")
    print(f"  Train top-1 : {top1*100:.1f}%")
    print(f"  LR          : {lr:.6f}")

    # ── Validation history ────────────────────────────────────────────────────
    if val_results:
        print(f"\n  VALIDATION HISTORY  (best: {best_top1*100:.2f}% @ epoch {best_epoch})")
        last_5 = sorted(val_results.keys())[-5:]
        for e in last_5:
            marker = " <-- BEST" if e == best_epoch else ""
            print(f"    Epoch {e:3d} :  val top-1 = {val_results[e]*100:.2f}%{marker}")

    # ── ETA estimate ──────────────────────────────────────────────────────────
    if last_train:
        epochs_left = TOTAL_EPOCHS - ep + 1
        sec_per_epoch = ITERS_EPOCH * 0.65   # ~0.65s per iter observed
        eta_sec = int(epochs_left * sec_per_epoch)
        eta_min = eta_sec // 60
        eta_sec_rem = eta_sec % 60
        print(f"\n  ETA (approx)  : {eta_min}m {eta_sec_rem}s")

    print(f"\n  Log : {log_path}")
    print(f"  Refreshing every {REFRESH_SEC}s  |  Ctrl+C to stop")
    print("=" * 65)


def main():
    print("Searching for training log...")
    while True:
        log_path = find_log()
        if log_path:
            break
        print("  Log not found yet — retrying in 3s...")
        time.sleep(3)

    print(f"Found log: {log_path}\nStarting monitor...\n")
    time.sleep(1)

    try:
        while True:
            display(log_path)
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
