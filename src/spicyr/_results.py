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

    def signif_plot(self, fdr: bool = False, breaks=None, comparison_group: str | None = None,
                    colours=("#4575B4", "white", "#D73027"), marks_to_plot=None, cutoff: float = 0.05, ax=None):
        """Bubble plot of every pair, as spicyR's ``signifPlot()``.

        Each pair is a disc at (to, from). Its left half is coloured by the excess in the reference condition and
        its right half by the excess in the comparison condition; the radius grows with -log10 p (the BH-adjusted p
        with ``fdr=True``), and a black
        ring marks p (or, with ``fdr=True``, the BH-adjusted p) below ``cutoff``. ``breaks`` is
        ``(low, high, step)`` for the colour scale. For survival results each disc is one colour: the log hazard
        ratio per SD.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Wedge

        t = self.cell_results
        if self.survival:
            pv, a, b = t["p_value"].to_numpy(float), t["log_hr_sd"].to_numpy(float), t["log_hr_sd"].to_numpy(float)
            fr_, to_ = t["from"].to_numpy(), t["to"].to_numpy()
        else:
            cond = [c for c in self.p_value.columns if c.startswith("condition")]
            coef = cond[0] if comparison_group is None else f"condition{comparison_group}"
            if coef not in cond:
                raise ValueError(f"comparison_group must be one of {[c[len('condition'):] for c in cond]}")
            cf = self.coefficient
            pv = self.p_value[coef].to_numpy(float)
            a = cf["(Intercept)"].to_numpy(float)                 # reference condition
            b = a + cf[coef].to_numpy(float)                      # comparison condition
            ft = self.p_value.index.to_series().str.split("__", n=1, expand=True)
            fr_, to_ = ft[0].to_numpy(), ft[1].to_numpy()
        if fdr:
            pv = p_adjust_bh(pv)
        sig = pv < cutoff
        size = -np.log10(pv)
        marks = sorted(set(fr_) | set(to_)) if marks_to_plot is None else list(marks_to_plot)
        keep = np.isin(fr_, marks) & np.isin(to_, marks)
        xs = sorted(set(to_[keep])); ys = sorted(set(fr_[keep]))
        xi = {v: k for k, v in enumerate(xs)}; yi = {v: k for k, v in enumerate(ys)}

        vals = np.concatenate([a[keep], b[keep]])
        if breaks is None:
            lo, hi = np.round(np.nanmin(vals), 1), np.round(np.nanmax(vals), 1)
            ticks = np.linspace(lo, hi, 6)
        else:
            lo, hi = breaks[0], breaks[1]
            ticks = np.arange(lo, hi + breaks[2] / 2, breaks[2])
        lo, hi = min(lo, -1e-9), max(hi, 1e-9)                   # the scale is centred on 0
        cmap = LinearSegmentedColormap.from_list("spicy", list(colours))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)
        smax = np.nanmax(size[keep]) if keep.any() else 1.0

        if ax is None:
            fig = plt.figure(figsize=(0.42 * len(xs) + 2.6, 0.42 * len(ys) + 1.6), layout="constrained")
            ax = fig.gca()
        for k in np.flatnonzero(keep):
            if not np.isfinite(size[k]):
                continue
            x, y = xi[to_[k]], yi[fr_[k]]
            rad = max(size[k] / smax / 2, 0.15)
            ca = cmap(norm(np.clip(a[k], lo, hi))); cb = cmap(norm(np.clip(b[k], lo, hi)))
            ax.add_patch(Wedge((x, y), rad, 90, 270, facecolor=ca, edgecolor="none"))     # left: reference
            ax.add_patch(Wedge((x, y), rad, -90, 90, facecolor=cb, edgecolor="none"))     # right: comparison
            if sig[k]:
                ax.add_patch(Circle((x, y), rad, fill=False, edgecolor="black", linewidth=1))
        ax.set_xlim(-0.6, len(xs) - 0.4); ax.set_ylim(-0.6, len(ys) - 0.4)
        ax.set_aspect("equal")
        ax.set_xticks(range(len(xs)), xs, rotation=45, ha="right"); ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel("Cell type j (to)"); ax.set_ylabel("Cell type i (from)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
        labels = [f"{v:.3g}" for v in ticks]
        labels[0], labels[-1] = "avoidance", "attraction"
        cbar.set_ticklabels(labels)
        cbar.set_label("log HR per SD" if self.survival else "Localisation (excess)")

        # legends: significance ring, -log10 p sizes, and (two conditions) which half is which
        handles = [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=10,
                          label=("BH-adjusted p" if fdr else "p-value") + f" < {cutoff}")]
        for q in np.unique(np.round(np.linspace(smax / 4, smax, 3), 1)):
            handles.append(Line2D([], [], marker="o", ls="", color="grey", alpha=0.6,
                                  markersize=2 * 18 * max(q / smax / 2, 0.15), label=f"{q:g}"))
        fig = ax.figure
        leg = fig.legend(handles=handles[1:], title="-log10 adjusted p" if fdr else "-log10 p", loc="upper left", bbox_to_anchor=(1.0, 0.95),
                         frameon=False, fontsize=8, title_fontsize=9, labelspacing=1.2, borderpad=0.2)
        fig.legend(handles=handles[:1], loc="upper left", bbox_to_anchor=(1.0, 0.55), frameon=False, fontsize=8)
        if not self.survival:
            # which half is which condition: half-disc symbols, as spicyR's legend
            from matplotlib.legend_handler import HandlerBase

            class _Half(HandlerBase):
                def __init__(self, left):
                    super().__init__()
                    self.left = left

                def create_artists(self, legend, orig, x0, y0, width, height, fontsize, trans):
                    r = 0.45 * height
                    th = (90, 270) if self.left else (-90, 90)
                    return [Wedge((x0 + width / 2, y0 + height / 2), r, *th, facecolor="grey", transform=trans)]

            ref, comp = self.levels[0], coef[len("condition"):]
            hl, hr = Line2D([], []), Line2D([], [])
            fig.legend(handles=[hl, hr], labels=[ref, comp], handler_map={hl: _Half(True), hr: _Half(False)},
                       title="Condition", loc="upper left", bbox_to_anchor=(1.0, 0.42), frameon=False,
                       fontsize=8, title_fontsize=9, handleheight=1.6, handlelength=1.6)
        return ax

    def __repr__(self):
        t = self.cell_results
        n = len(t[["from", "to"]].drop_duplicates())
        what = "association with survival" if self.survival else "; ".join(f"{l} vs {self.levels[0]}" for l in self.levels[1:])
        scale = f"k = {self.k} nearest neighbours" if self.k is not None else "r = " + ", ".join(f"{v:g}" for v in self.r)
        lines = [f"spicyr (cell-level test): {n} pairs, {what}, {scale}"]
        if self.subject is None:
            lines.append(f"Units: {len(self.image_ids)} images (no subject given: each image is a patient)")
        else:
            lines.append(f"Units: {len(set(self.subject))} patients with {len(self.image_ids)} images")
        sig = f"BH-adjusted p < 0.05: {int((t['p_adj'] < 0.05).sum())} pairs"
        if "adjusted_p_adj" in t.columns:
            sig += f" ({int((t['adjusted_p_adj'] < 0.05).sum())} after adjusting for abundance)"
        lines += [sig, "See top_pairs() and .cell_results."]
        return "\n".join(lines)
