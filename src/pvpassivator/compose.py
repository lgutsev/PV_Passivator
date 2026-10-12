"""Multi-panel figures from house-style renders (matplotlib; run in a separate process from OVITO).

``compose_panels`` tiles PNG renders with bold panel letters, optional titles, one element legend in the render
colours, and an optional scale bar for panels whose pixel scale is known. It never resamples the renders beyond
matplotlib's display scaling, so every panel keeps the camera and scale it was rendered with.
"""
from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

from .render import COLORS, FOCUS_COLOR


def compose_panels(images: list[str | Path], out: str | Path, ncols: int = 2, titles: list[str] | None = None,
                   letters: bool = True, legend: list[str] | None = None, focus_label: str | None = None,
                   scale_bar: dict | None = None, panel_width_in: float = 3.3, dpi: int = 300) -> Path:
    """Write a multi-panel figure.

    ``legend``: element symbols to list, drawn as discs in the render colours; ``focus_label`` adds a magenta disc
    with that text (e.g. "I (oxidised)"). ``scale_bar``: {"px_per_A": float, "length_A": float, "panels": [indices]}
    draws a bar of that length on the listed panels (default all); pass the px/A of the renders, never a guess.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    images = [Path(p) for p in images]
    if not images:
        raise ValueError("no images")
    if titles is not None and len(titles) != len(images):
        raise ValueError("titles must match images")
    arrays = [plt.imread(str(p)) for p in images]
    ncols = max(1, min(ncols, len(arrays)))
    nrows = -(-len(arrays) // ncols)
    aspect = max(a.shape[0] / a.shape[1] for a in arrays)
    fig_w = panel_width_in * ncols
    legend_h = 0.35 if (legend or focus_label) else 0.0
    fig_h = panel_width_in * aspect * nrows + 0.25 * nrows * bool(titles) + legend_h
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    bar_panels = set(range(len(arrays))) if scale_bar and "panels" not in scale_bar else set((scale_bar or {}).get("panels", []))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(arrays):
            continue
        img = arrays[k]
        ax.imshow(img, interpolation="lanczos")
        if titles:
            ax.set_title(titles[k], fontsize=9, pad=3)
        if letters:
            ax.text(0.01, 0.99, ascii_lowercase[k], transform=ax.transAxes, fontsize=12, fontweight="bold",
                    va="top", ha="left")
        if scale_bar and k in bar_panels:
            L = scale_bar["px_per_A"] * scale_bar["length_A"]
            h, w = img.shape[:2]
            x0, y0 = w * 0.96 - L, h * 0.94
            ax.plot([x0, x0 + L], [y0, y0], color="black", lw=2.0, solid_capstyle="butt")
            ax.text(x0 + L / 2, y0 - h * 0.02, f"{scale_bar['length_A']:g} Å", ha="center", va="bottom", fontsize=8)
    handles = []
    for e in legend or []:
        handles.append(Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=COLORS.get(e, (0.6, 0.6, 0.6)),
                              markeredgecolor=(0.3, 0.3, 0.3), markeredgewidth=0.5, label=e))
    if focus_label:
        handles.append(Line2D([], [], ls="", marker="o", markersize=9, markerfacecolor=FOCUS_COLOR,
                              markeredgecolor=(0.3, 0.3, 0.3), markeredgewidth=0.5, label=focus_label))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8,
                   handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, legend_h / fig_h, 1, 1), pad=0.3)
    out = Path(out)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
