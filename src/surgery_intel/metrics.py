"""Phase-recognition metrics: accuracy, precision, recall, Jaccard, F1 + relaxed tolerance."""
import numpy as np


def _per_phase_scores(gt, pred, num_phases):
    """Per-phase scores, omitting phases absent from gt (TUNeS/SurgFM protocol)."""
    prec, rec, jac = [], [], []
    for p in range(num_phases):
        if not (gt == p).any():
            continue  # phase not annotated in this video: omit, don't zero-fill
        tp = int(((pred == p) & (gt == p)).sum())
        fp = int(((pred == p) & (gt != p)).sum())
        fn = int(((pred != p) & (gt == p)).sum())
        prec.append(tp / (tp + fp) if tp + fp else 0.0)
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
        jac.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)
    return prec, rec, jac


def relaxed_correct(gt, pred, tol_frames):
    """Frames within tol_frames of a true phase boundary count as correct either way."""
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if tol_frames <= 0:
        return pred == gt
    boundaries = set(np.where(np.diff(gt) != 0)[0].tolist())
    near = np.zeros_like(gt, dtype=bool)
    for b in boundaries:
        lo = max(0, b - tol_frames + 1)
        hi = min(len(gt), b + tol_frames + 1)
        near[lo:hi] = True
    ok = pred == gt
    ok[near] = True  # tolerate transition wobble (standard relaxed protocol)
    return ok


def video_scores(gt, pred, num_phases=7, tol_frames=0):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if tol_frames:
        acc = float(relaxed_correct(gt, pred, tol_frames).mean())
    else:
        acc = float((pred == gt).mean())
    prec, rec, jac = _per_phase_scores(gt, pred, num_phases)
    mp, mr = float(np.mean(prec)) if prec else 0.0, float(np.mean(rec)) if rec else 0.0
    f1 = 2 * mp * mr / (mp + mr) if mp + mr else 0.0  # harmonic mean of means
    return {
        "accuracy": acc,
        "precision": mp,
        "recall": mr,
        "jaccard": float(np.mean(jac)) if jac else 0.0,
        "f1": f1,
    }
