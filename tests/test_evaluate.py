import csv
import sys

from surgery_intel import evaluate


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "phase"])
        w.writerows(rows)


def test_evaluate_strict_and_relaxed(tmp_path, capsys, monkeypatch):
    man = str(tmp_path / "manifest.csv")
    pred = str(tmp_path / "preds.csv")
    rows = [["41", i, 0 if i < 4 else 1] for i in range(8)]
    _write(man, rows)
    _write(pred, rows)  # perfect preds
    monkeypatch.setattr(sys, "argv", ["evaluate", "--preds", pred,
                                      "--manifest", man, "--relaxed-sec", "10"])
    evaluate.main()
    out = capsys.readouterr().out
    assert "== strict" in out and "== relaxed" in out
    assert "100.00" in out
