"""calibration_check(): shuffles the condition across patients and reports the share of small p-values."""

import numpy as np
import pytest

import spicyr


@pytest.fixture(scope="module")
def cells():
    return spicyr.datasets.example_cells()


def test_same_as_shuffling_by_hand(cells):
    out = spicyr.calibration_check(cells, condition="condition", subject="patient", r=30, n_shuffles=2, rng=1)
    assert list(out.columns) == ["shuffle", "threshold", "percent"]
    assert out["shuffle"].tolist() == [1, 1, 2, 2]
    assert out["threshold"].tolist() == [0.05, 0.01, 0.05, 0.01]
    # by hand: each patient keeps one condition, moved as a whole
    rng = np.random.default_rng(1)
    first = ~cells["patient"].duplicated()
    units, labels = cells.loc[first, "patient"].to_numpy(), cells.loc[first, "condition"].to_numpy()
    lab = dict(zip(units, labels[rng.permutation(len(units))], strict=True))
    d = cells.assign(shuffled=cells["patient"].map(lab))
    p = spicyr.spicy(d, condition="shuffled", subject="patient", r=30).cell_results["p_value"].dropna()
    assert out["percent"].iloc[:2].tolist() == pytest.approx([100 * (p < 0.05).mean(), 100 * (p < 0.01).mean()])
    assert "shuffledCondition0" not in cells.columns  # the input is left as it was


def test_missing_condition_stays_missing_and_survival(cells):
    d = cells.copy()
    d.loc[d["patient"] == d["patient"].iloc[0], "condition"] = np.nan
    out = spicyr.calibration_check(d, condition="condition", subject="patient", r=30, n_shuffles=1, rng=2)
    assert len(out) == 2
    surv = spicyr.calibration_check(cells, survival=("time", "event"), subject="patient", r=30, n_shuffles=1, rng=3)
    assert len(surv) == 2 and surv["percent"].between(0, 100).all()


def test_errors(cells):
    with pytest.raises(ValueError, match="not both"):
        spicyr.calibration_check(cells, subject="patient", r=30)
    with pytest.raises(ValueError, match="columns not found"):
        spicyr.calibration_check(cells, condition="nothere", subject="patient", r=30)
    with pytest.raises(TypeError, match="in memory"):
        spicyr.calibration_check("cells.zarr", condition="condition", r=30)


def test_plot_image_without_a_pair(cells):
    import matplotlib

    matplotlib.use("Agg")
    ax = spicyr.plot_image(cells, cells["imageID"].iloc[0])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == sorted(cells.loc[cells["imageID"] == cells["imageID"].iloc[0], "cellType"].astype(str).unique())
    with pytest.raises(ValueError, match="both"):
        spicyr.plot_image(cells, cells["imageID"].iloc[0], from_="tumour")
    with pytest.raises(ValueError, match="circles"):
        spicyr.plot_image(cells, cells["imageID"].iloc[0], r=10)
