"""Generate demo GT/preds CSVs so the viewer works without gated Cholec80 data."""
import csv
import os
import random

PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def main():
    random.seed(7)
    os.makedirs("outputs", exist_ok=True)
    # 120 frames, phase blocks with a couple of tricky boundaries
    gt = []
    blocks = [(0, 20, 0), (20, 45, 1), (45, 60, 2), (60, 85, 3), (85, 95, 4), (95, 110, 5), (110, 120, 6)]
    for lo, hi, p in blocks:
        gt.extend([p] * (hi - lo))
    pred = []
    for i, g in enumerate(gt):
        r = random.random()
        if 18 <= i <= 22 or 58 <= i <= 62:
            # boundary wobble: early/late by 1-2 frames
            pred.append(gt[max(0, i - 2)] if r < 0.5 else g)
        elif r < 0.06:
            pred.append(random.choice([x for x in range(7) if x != g]))
        else:
            pred.append(g)
    with open("outputs/demo_gt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "phase"])
        for i, g in enumerate(gt):
            w.writerow([i, g])
    with open("outputs/demo_preds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "phase"])
        for i, p in enumerate(pred):
            w.writerow([i, p])
    print(f"wrote outputs/demo_gt.csv + outputs/demo_preds.csv ({len(gt)} frames)")


if __name__ == "__main__":
    main()
