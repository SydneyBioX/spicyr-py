"""The results of ``spicy()``, with the accessors of spicyR's ``SpicyResults`` (topPairs, bind, the plots)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._cell import p_adjust_bh


@dataclass
class SpicyResults:
    """Results of :func:`spicyr.spicy`.

    ``cell_results`` is the full table, one row per pair (and per level when there are more than two
    conditions): the excess in the reference condition (``excess_ref``) and the comparison condition, the
    difference, its standard error, Satterthwaite df, p-value and BH-adjusted p-value, the frailty variance
    ``tau2``, and the availability-adjusted difference (``adjusted_*``). For survival: the score test
    (``score_coefficient``, ``p_value``) and the hazard ratio per SD of the shrunken excess.
    """

    cell_results: pd.DataFrame
    levels: list | None
    survival: bool
    radius_results: pd.DataFrame | None = None
    image_ids: list = field(default_factory=list)
    condition: list | None = None
    subject: list | None = None
    pairwise_assoc: dict = field(default_factory=dict)
    r: list | None = None
    k: int | None = None
    method: str = "cell"

    # --- spicyR-style matrices -----------------------------------------------------------------------
    def _wide(self, col):
        t = self.cell_results
        if "level" in t.columns:
            key = t["from"] + "__" + t["to"]
            labels = list(dict.fromkeys(key))
            out = pd.DataFrame(index=labels)
            out["(Intercept)"] = t["excess_ref"].groupby(key).first().reindex(labels) if col == "excess_difference" else np.nan
            for l in self.levels[1:]:
                k = (t["level"] == l).to_numpy()
                out[f"condition{l}"] = pd.Series(t.loc[k, col].to_numpy(), index=key[k]).reindex(labels)
            return out
        term = "condition" if self.survival else f"condition{self.levels[1]}"
        first = (t["excess_ref"] if col == "excess_difference" and not self.survival else np.nan)
        return pd.DataFrame({"(Intercept)": first, term: t[col]}, index=t.index)

    @property
    def coefficient(self) -> pd.DataFrame:
        return self._wide("log_hr_sd" if self.survival else "excess_difference")

    @property
    def p_value(self) -> pd.DataFrame:
        return self._wide("p_value")

    @property
    def se(self) -> pd.DataFrame:
        return self._wide("score_se" if self.survival else "se")

    # --- accessors -----------------------------------------------------------------------------------
    def top_pairs(self, n: int | None = 10, coef: str | None = None, adj: str = "fdr", cutoff: float | None = None,
                  figures: int | None = None) -> pd.DataFrame:
        """The most significant pairs, as spicyR's ``topPairs()``."""
        pv, cf = self.p_value, self.coefficient
        if coef is None:
            coef = [c for c in pv.columns if c.startswith("condition")][0]
        p = pv[coef].to_numpy(float)
        if adj not in ("fdr", "BH"):
            raise ValueError("only adj = 'fdr' (Benjamini-Hochberg) is available.")
        res = pd.DataFrame({"intercept": cf["(Intercept)"].to_numpy(), "coefficient": cf[coef].to_numpy(),
                            "p.value": p, "adj.pvalue": p_adjust_bh(p)}, index=pv.index)
        ft = res.index.to_series().str.split("__", n=1, expand=True)
        res["from"], res["to"] = ft[0].to_numpy(), ft[1].to_numpy()
        res = res.sort_values("p.value", kind="stable")
        if cutoff is not None:
            res = res[res["adj.pvalue"] <= cutoff]
        if n is not None:
            res = res.head(n)
        if figures is not None:
            res = res.apply(lambda c: c.map(lambda v: float(f"{v:.{figures}g}")) if c.dtype.kind == "f" else c)
        return res

    def bind(self, pair: str | None = None) -> pd.DataFrame:
        """Per-image excess of every pair (or one ``"from__to"`` pair) with the image's condition and subject."""
        out = pd.DataFrame({"imageID": self.image_ids})
        if self.condition is not None:
            out["condition"] = self.condition
        if self.subject is not None:
            out["subject"] = self.subject
        keys = [pair] if pair is not None else list(self.pairwise_assoc)
        for k in keys:
            out[k] = self.pairwise_assoc[k]
        return out

    def box_plot(self, from_: str, to: str, ax=None):
        """Per-image excess of one pair by condition (spicyR's ``spicyBoxPlot()``)."""
        import matplotlib.pyplot as plt

        d = self.bind(f"{from_}__{to}")
        ax = ax or plt.figure(figsize=(4, 4)).gca()
        groups = [g for g in (self.levels or [])]
        data = [d.loc[d["condition"] == g, f"{from_}__{to}"].dropna().to_numpy() for g in groups]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(groups) + 1), groups)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Condition")
        ax.set_ylabel(f"Excess (extra {to} cells per {from_} cell)")
        ax.set_title(f"{to} cells around {from_} cells")
        return ax

    def signif_plot(self, cutoff: float = 0.05, ax=None):
        """Bubble plot of every pair: colour = difference in excess, size = -log10 p, ring = BH < cutoff
        (spicyR's ``signifPlot()``)."""
        import matplotlib.pyplot as plt

        t = self.cell_results
        if "level" in t.columns:
            t = t[t["level"] == self.levels[1]]
        fr = sorted(t["from"].unique()); to = sorted(t["to"].unique())
        ax = ax or plt.figure(figsize=(0.45 * len(to) + 2.5, 0.45 * len(fr) + 1.5)).gca()
        x = t["to"].map({v: i for i, v in enumerate(to)}); y = t["from"].map({v: i for i, v in enumerate(fr)})
        col = t["log_hr_sd"] if self.survival else t["excess_difference"]
        lim = np.nanmax(np.abs(col)) or 1
        sc = ax.scatter(x, y, c=col, cmap="RdBu_r", vmin=-lim, vmax=lim, s=20 + 40 * -np.log10(t["p_value"].clip(1e-10)),
                        edgecolors=np.where(t["p_adj"] < cutoff, "black", "none"), linewidths=1.2)
        ax.set_xticks(range(len(to)), to, rotation=90); ax.set_yticks(range(len(fr)), fr)
        ax.set_xlabel("to (counted)"); ax.set_ylabel("from (centre)")
        plt.colorbar(sc, ax=ax, label="log HR per SD" if self.survival else "difference in excess")
        return ax

    def __repr__(self):
        n = len(self.cell_results)
        what = "survival" if self.survival else f"conditions {self.levels}"
        sig = int((self.cell_results["p_adj"] < 0.05).sum())
        return f"SpicyResults (spicyR Cell): {n} tests, {what}; {sig} with BH-adjusted p < 0.05"
