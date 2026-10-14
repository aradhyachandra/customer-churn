"""C6 helpers: cohort medians skip the censored first cohort and incomplete cells; report config
settings each have a rationale."""

import pandas as pd
import pytest

from churn.config import load_config
from churn.run_c6 import cohort_medians


def test_cohort_medians_skip_first_cohort_and_cells_at_or_after_the_last_month():
    rates = pd.DataFrame({"1": [0.9, 0.2, 0.4, 0.1], "3": [0.9, 0.3, 0.5, None]},
                         index=["2011-01", "2011-02", "2011-03", "2011-04"])
    out = cohort_medians(rates, "2011-06", (1, 3), skip_first=True)
    # Month 1: cohorts Feb, Mar, Apr (Apr + 1 = May < Jun) -> median of 0.2, 0.4, 0.1.
    assert out["m1"] == {"median": pytest.approx(0.2), "cohorts": 3, "min": 0.1, "max": 0.4}
    # Month 3: Feb + 3 = May kept; Mar + 3 = Jun is the partial last month, dropped.
    assert out["m3"]["cohorts"] == 1 and out["m3"]["median"] == pytest.approx(0.3)


def test_every_report_setting_has_a_rationale():
    cfg = load_config("report")
    for key in cfg:
        if not key.endswith("_rationale"):
            assert cfg.get(f"{key}_rationale"), f"{key} has no rationale"


def test_segment_bands_have_one_label_per_band():
    for name, spec in load_config("report")["segments"].items():
        if "edges" in spec:
            assert len(spec["labels"]) == len(spec["edges"]) - 1, name
