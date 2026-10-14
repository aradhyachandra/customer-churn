# C6: Clean-clone reproducibility check

*Checked 2026-10-14. All numbers below come from `results/c6_clean_clone_check.json`, which was
written from the check's own output.*

## Result: everything reproduces

A fresh copy of the repository, with a fresh virtual environment, rebuilt every result from the
raw file and matched the committed results.

- **Results files:** 35 of 35 match. 26 are byte-identical, including all 7 figures and every
  CSV. 9 JSON files are identical once their run information is ignored (`generated_at` and the
  `git` block, plus `validation_generated_at` in `c4_test_writeup.json`).
- **Models:** the refitted rule, logistic and boosting models have the same sha256 as the frozen
  manifest `results/c4_final_models.json` (boosting `2dfb635d852d480ad6c617432bea012d0d85285847b0affa3a0dcfe52a3ff146`).
- **Test snapshot:** its sha256 equals the one recorded by the one-shot test, so the `run_c5` and
  `run_c6` guards accepted it, and the rebuilt C5/C6 numbers are the same (for example the central
  campaign of 1,567 customers and £22,621).
- **Tests:** 202 unit tests and 14 real-data tests pass in the clone.
- **Docs:** every regenerated doc (C1 to C5, REPORT.md, README.md) differs from the working copy
  only in its "generated on ... (git ...)" banner.
- **`churn.run_c4_test` was not run**, either in the clone or in the working copy.
  `results/c4_test_metrics.json` came from the repository and is byte-identical afterwards.

## What was done

1. `git clone` of the local repository at HEAD `7d3f0b8` into a temporary folder.
2. **Overlay:** C6 is not committed yet, so the uncommitted working-tree changes (26 entries in
   `git status`, new folders counted once; nothing gitignored) were copied over the clone. The
   check therefore tests the repository as it will be once C6 is committed. Rerunning it after the
   commit would make it a pure clone. Written after the check, and so not part of it: this note,
   `results/c6_clean_clone_check.json` and the C6 entries in NOTES.md and PROGRESS.md.
3. A fresh venv with Python 3.13.5: `pip install -r requirements.txt` and `pip install -e .`.
   `pip check` found no broken requirements.
4. The raw file was copied into `data/raw/` (no re-download). The loader verified its sha256
   (`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`).
5. Ran, in order: `run_c1`, `run_c2`, `run_c3`, `run_c4`, `render_c4_test`, `run_c5`, `run_c6`.
6. Ran `pytest` and `pytest -m realdata`.
7. Compared the clone's `results/` with the committed-plus-C6 `results/` using
   `python -m churn.compare_runs` (JSON run information ignored, everything else byte for byte).

## Time and memory (8 GB MacBook Air)

| Step | Seconds | Peak memory (MB) |
|---|---:|---:|
| run_c1 | 25.58 | 978 |
| run_c2 | 11.59 | 536 |
| run_c3 | 5.98 | 722 |
| run_c4 | 37.96 | 270 |
| render_c4_test | 1.84 | 237 |
| run_c5 | 9.75 | 253 |
| run_c6 | 1.91 | 272 |

Steps run one after another, so the peak is about 1 GB (reading the Excel file in `run_c1`).

## One thing to know: why streamlit is pinned at 1.58.0

Newer streamlit releases need pyarrow below 26. pyarrow 25 writes the parquet snapshots with
different bytes (the writer version is stored in the file), so the test snapshot's sha256 would
change and `run_c5`/`run_c6` would refuse to run. Streamlit 1.58.0 is the newest release that
accepts the pinned pyarrow 26.0.0. This check confirms that the combination reproduces everything.
