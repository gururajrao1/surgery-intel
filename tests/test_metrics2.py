import numpy as np
from surgery_intel.metrics import video_scores


def test_perfect_scores():
    gt = np.array([0, 0, 1, 1, 2, 2])
    s = video_scores(gt, gt, num_phases=3)
    assert s["accuracy"] == 1.0
    assert s["jaccard"] == 1.0


def test_relaxed_forgives_boundary():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1])
    strict = video_scores(gt, pred, tol_frames=0)["accuracy"]
    relaxed = video_scores(gt, pred, tol_frames=2)["accuracy"]
    assert relaxed >= strict
    assert relaxed == 1.0


def test_partial():
    gt = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0, 0, 0])
    s = video_scores(gt, pred, num_phases=2)
    assert 0.0 < s["accuracy"] < 1.0


def test_absent_phases_omitted_not_zero_filled():
    # Only phases 0,1 appear; perfect prediction must score 1.0 even with num_phases=7
    gt = np.array([0, 0, 1, 1])
    s = video_scores(gt, gt, num_phases=7)
    assert s["jaccard"] == 1.0 and s["precision"] == 1.0 and s["f1"] == 1.0


def test_f1_is_harmonic_mean_of_means():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    s = video_scores(gt, pred, num_phases=2)
    mp, mr = s["precision"], s["recall"]
    assert s["f1"] == 2 * mp * mr / (mp + mr)
