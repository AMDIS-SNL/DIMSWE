#!/usr/bin/env python3
"""Create the deterministic DIMSWE ground-truth report figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


FIXED_DATE = datetime(2026, 8, 29, tzinfo=timezone.utc)
COLORS = {
    "test2a": "#3B6FB6",
    "test2b": "#D05A3A",
    "cloud": "#3268A8",
    "rain": "#B34D70",
    "onset": "#E69F00",
    "sustain": "#009E73",
    "mature": "#7A5195",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(
        png,
        dpi=300,
        metadata={"Software": "DIMSWE deterministic ground-truth figure script"},
    )
    fig.savefig(
        pdf,
        metadata={
            "Creator": "DIMSWE deterministic ground-truth figure script",
            "CreationDate": FIXED_DATE,
            "ModDate": FIXED_DATE,
        },
    )
    plt.close(fig)
    return (png, pdf)


def panel_labels(axes, fontsize=10):
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.text(
            0.012,
            0.988,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            zorder=20,
        )


def add_event_lines(axes, labels=False):
    events = (
        (5100.0 / 3600.0, "onset", COLORS["onset"]),
        (6100.0 / 3600.0, "sustained", COLORS["sustain"]),
        (12000.0 / 3600.0, "mature", COLORS["mature"]),
    )
    for axis in np.asarray(axes).reshape(-1):
        for value, label, color in events:
            axis.axvline(value, color=color, lw=0.9, ls="--", alpha=0.85)
            if labels:
                axis.text(
                    value,
                    0.98,
                    label,
                    rotation=90,
                    color=color,
                    transform=axis.get_xaxis_transform(),
                    va="top",
                    ha="right",
                    fontsize=7,
                )


def initial_fields(n=401):
    lx = ly = 5.0e6
    g = 9.80616
    f = 6.147e-5
    h0 = 750.0
    dh = 75.0
    sigx = sigy = 3.0 * lx / 40.0
    centers = ((0.4 * lx, 0.4 * ly), (0.6 * lx, 0.6 * ly))
    x = np.linspace(0.0, lx, n, endpoint=False)
    y = np.linspace(0.0, ly, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    gaussians = []
    xtildes = []
    ytildes = []
    cos2x = []
    cos2y = []
    for xc, yc in centers:
        tx = np.pi * (xx - xc) / lx
        ty = np.pi * (yy - yc) / ly
        xp = lx / (np.pi * sigx) * np.sin(tx)
        yp = ly / (np.pi * sigy) * np.sin(ty)
        xt = lx / (2.0 * np.pi * sigx) * np.sin(2.0 * tx)
        yt = ly / (2.0 * np.pi * sigy) * np.sin(2.0 * ty)
        gaussians.append(np.exp(-0.5 * (xp * xp + yp * yp)))
        xtildes.append(xt)
        ytildes.append(yt)
        cos2x.append(np.cos(2.0 * tx))
        cos2y.append(np.cos(2.0 * ty))
    correction = 4.0 * np.pi * sigx * sigy / (lx * ly)
    height = h0 - dh * (gaussians[0] + gaussians[1] - correction)
    hx = dh / sigx * sum(xt * gg for xt, gg in zip(xtildes, gaussians))
    hy = dh / sigy * sum(yt * gg for yt, gg in zip(ytildes, gaussians))
    u = -(g / f) * hy
    v = (g / f) * hx
    dvdx = g * dh / (f * sigx**2) * sum(
        (cc - xt * xt) * gg
        for cc, xt, gg in zip(cos2x, xtildes, gaussians)
    )
    dudy = -g * dh / (f * sigy**2) * sum(
        (cc - yt * yt) * gg
        for cc, yt, gg in zip(cos2y, ytildes, gaussians)
    )
    vorticity = dvdx - dudy
    c, a, d = 0.05, 1.0 / 3.0, 0.5 * lx
    s_over_g_anomaly = c * np.exp(
        -((xx - 0.5 * lx) ** 2 + (yy - 0.5 * ly) ** 2) / (a * a * d * d)
    )
    return x / 1000.0, y / 1000.0, height, u, v, vorticity, s_over_g_anomaly


def figure1(output: Path):
    x, y, h, u, v, zeta, thermal = initial_fields()
    extent = (0, 5000, 0, 5000)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), constrained_layout=True)
    im = axes[0].imshow(
        h - 750.0, origin="lower", extent=extent, cmap="viridis", vmin=-70, vmax=6
    )
    stride = 5
    axes[0].streamplot(
        x[::stride], y[::stride], u[::stride, ::stride], v[::stride, ::stride],
        density=0.72, color="white", linewidth=0.65, arrowsize=0.72,
    )
    axes[0].plot((2000, 3000), (2000, 3000), "w+", ms=9, mew=1.7)
    colorbar = fig.colorbar(im, ax=axes[0], shrink=0.82, pad=0.025)
    colorbar.set_label(r"$h-H_0$ (m)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    axes[0].set_title("Depth anomaly and\ngeostrophic flow", fontsize=14, linespacing=1.05)

    limit = float(np.max(np.abs(zeta * 1.0e5)))
    im = axes[1].imshow(
        zeta * 1.0e5,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    colorbar = fig.colorbar(im, ax=axes[1], shrink=0.82, pad=0.025)
    colorbar.set_label(r"relative vorticity ($10^{-5}$ s$^{-1}$)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    axes[1].set_title("Analytical relative vorticity", fontsize=14)

    im = axes[2].imshow(
        100.0 * thermal, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=5.0
    )
    colorbar = fig.colorbar(im, ax=axes[2], shrink=0.82, pad=0.025)
    colorbar.set_label(r"$100(b/g-1)$ (%)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    axes[2].set_title("Thermal/buoyancy anomaly", fontsize=14)

    for axis in axes:
        axis.set_xlabel("x (km)", fontsize=13)
        axis.set_ylabel("y (km)", fontsize=13)
        axis.set_aspect("equal")
        axis.set_xticks((0, 2500, 5000))
        axis.set_yticks((0, 2500, 5000))
        axis.tick_params(labelsize=11)
    panel_labels(axes, fontsize=15)
    paths = save_figure(fig, output / "figure1_doublevortex_initial_state")
    return paths, {
        "panels": ["h-H0 with streamlines", "relative vorticity", "100*(b/g-1)"],
        "state_indices": [0],
        "times_s": [0.0],
        "variables_units": {"h-H0": "m", "velocity": "m s-1", "relative_vorticity": "1e-5 s-1", "100*(b/g-1)": "%"},
        "color_limits": {"h-H0_m": [-70, 6], "thermal_percent": [0, 5], "vorticity_1e5_s-1": [-limit, limit]},
    }


def figure2(a, b, output: Path):
    from matplotlib.lines import Line2D

    ta = a["time_s"] / 3600.0
    tb = b["time_s"] / 3600.0
    fig, axes = plt.subplots(
        2, 3, figsize=(15.6, 8.8), constrained_layout=True, sharex=True
    )

    test_lines = (
        (a, ta, COLORS["test2a"], "Test 2A"),
        (b, tb, COLORS["test2b"], "Test 2B"),
    )

    ax = axes[0, 0]
    for data, time, color, label in test_lines:
        ax.plot(time, data["Qc_mass"] / 1e12, color=color, lw=2.4, label=label)
    ax.set_ylabel(r"$M_c$ ($10^{12}$ m$^3$)", fontsize=13)
    ax.set_title("Integrated cloud water", fontsize=14)

    ax = axes[0, 1]
    for data, time, color, label in test_lines:
        exceedance = 1000.0 * (data["specific_Qc_maximum"] - 1.0e-4)
        ax.plot(time, exceedance, color=color, lw=2.4, label=label)
    ax.axhline(0.0, color="0.15", lw=1.2)
    ax.set_ylabel(
        r"$\max_{\boldsymbol{x}}(q_c-q_{\rm precip})$ (g kg$^{-1}$)",
        fontsize=13,
    )
    ax.set_title("Cloud-water threshold exceedance", fontsize=14)
    ax.text(
        0.97,
        0.08,
        "negative: entire domain below threshold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="0.25",
    )

    ax = axes[0, 2]
    ax.axhline(0.0, color="0.15", lw=0.8, zorder=0)
    ax.plot(
        tb,
        b["rain_water_mass"] / 1e8,
        color=COLORS["test2b"],
        lw=2.5,
        zorder=2,
    )
    ax.plot(
        ta,
        a["rain_water_mass"] / 1e8,
        color=COLORS["test2a"],
        lw=2.6,
        ls="--",
        zorder=3,
    )
    ax.set_ylabel(r"$M_r$ ($10^8$ m$^3$)", fontsize=13)
    ax.set_title("Integrated rain water", fontsize=14)
    ax.text(
        0.04,
        0.91,
        r"Test 2A: $M_r=0$ throughout",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["test2a"],
    )

    ax = axes[1, 0]
    amin = b["A_min_s-1"] * 3.6e6
    amax = b["A_max_s-1"] * 3.6e6
    ax.fill_between(
        tb,
        amin,
        amax,
        color="#6A8CAF",
        alpha=0.24,
        label=r"domain min--max $A$",
    )
    ax.plot(tb, amin, color="#315A82", lw=1.5, label=r"$A_{\min}$")
    ax.plot(tb, amax, color="#A34F42", lw=1.5, label=r"$A_{\max}$")
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.set_yscale("symlog", linthresh=0.02, linscale=0.8)
    ax.set_ylabel(r"$A=E-C$ (g kg$^{-1}$ h$^{-1}$)", fontsize=13)
    ax.set_title("Domain min–max $A$ (Test 2B)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.9)

    ax = axes[1, 1]
    ax.plot(
        tb,
        b["rain_source_mass_rate"] / 1e3,
        color=COLORS["rain"],
        lw=2.5,
    )
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.set_ylabel(r"$\int_\Omega hR\,dA$ ($10^3$ m$^3$ s$^{-1}$)", fontsize=13)
    ax.set_title("Integrated rain-production rate", fontsize=14)

    ax = axes[1, 2]
    for data, time, color, label in test_lines:
        lower = 100.0 * (data["saturation_ratio_minimum"] - 1.0)
        upper = 100.0 * (data["saturation_ratio_maximum"] - 1.0)
        ax.plot(
            time,
            lower,
            color=color,
            lw=1.8,
            ls="--",
            label=f"{label} domain min",
        )
        ax.plot(
            time,
            upper,
            color=color,
            lw=2.0,
            ls="-",
            label=f"{label} domain max",
        )
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.plot(
        tb[0],
        100.0 * (b["saturation_ratio_maximum"][0] - 1.0),
        "o",
        color=COLORS["test2b"],
        ms=5.5,
        zorder=5,
    )
    ax.text(
        0.18,
        0.92,
        "Test 2B initial: approximately 6%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["test2b"],
    )
    ax.set_ylabel(r"$100(q_v/q_{\rm sat}-1)$ (%)", fontsize=13)
    ax.set_title("Domain saturation-departure range", fontsize=14)
    ax.legend(loc="lower left", fontsize=9.6, ncol=2, framealpha=0.9)

    add_event_lines(axes, labels=False)
    for ax in axes[1, :]:
        ax.set_xlabel("time (h)", fontsize=13)
    for ax in axes.reshape(-1):
        ax.grid(color="0.88", lw=0.7)
        ax.tick_params(labelsize=11.5)
        ax.set_xlim(ta[0], ta[-1])
    panel_labels(axes, fontsize=14)

    figure_handles = [
        Line2D([0], [0], color=COLORS["test2a"], lw=2.7),
        Line2D([0], [0], color=COLORS["test2b"], lw=2.7),
        Line2D([0], [0], color=COLORS["onset"], lw=1.6, ls="--"),
        Line2D([0], [0], color=COLORS["sustain"], lw=1.6, ls="--"),
        Line2D([0], [0], color=COLORS["mature"], lw=1.6, ls="--"),
    ]
    figure_labels = (
        "Test 2A",
        "Test 2B",
        "5100 s: first certifiable rain",
        "6100 s: sustained-rain certification",
        "12,000 s: peak integrated production",
    )
    fig.legend(
        figure_handles,
        figure_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=5,
        fontsize=10.8,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )

    paths = save_figure(fig, output / "figure2_combined_chronology_regime_comparison")
    return paths, {
        "state_indices": list(map(int, a["step"])),
        "times_s": [float(a["time_s"][0]), float(a["time_s"][-1])],
        "variables_units": {
            "Qc_mass": "m3",
            "max_qc_minus_qprecip": "g kg-1",
            "Qr_mass": "m3",
            "A_domain_min_max": "g kg-1 h-1",
            "integrated_hR": "m3 s-1",
            "saturation_departure_domain_min_max": "%",
        },
        "event_markers_s": [5100.0, 6100.0, 12000.0],
        "rain_active_fraction_included": False,
        "interpretation_limit": "Test 2A and Test 2B differ in both initial vapor loading and mesh resolution; physical-regime comparison only, not a grid-convergence study",
    }


def figure3(data, output: Path):
    steps = (0, 50, 61, 120, 160)
    labels = ("initial", "last pre-rain", "sustained certified", "peak integrated R", "final")
    indices = [int(np.flatnonzero(data["step"] == step)[0]) for step in steps]
    variables = (
        ("supersaturation_percent", r"$100(q_v/q_{sat}-1)$ (%)", "RdBu_r", -2.5, 6.1),
        ("specific_cloud_g_kg-1", r"$q_c$ (g kg$^{-1}$)", "Blues", 0.0, 0.11),
        ("specific_rain_ug_kg-1", r"$q_r$ ($\mu$g kg$^{-1}$)", "PuRd", 0.0, 130.0),
        ("R_ug_kg-1_h-1", r"$R$ ($\mu$g kg$^{-1}$ h$^{-1}$)", "magma", 0.0, 180.0),
    )
    fig, axes = plt.subplots(
        4, 5, figsize=(11.2, 8.4), constrained_layout=True, sharex=True, sharey=True
    )
    extent = (0, 5000, 0, 5000)
    vort_levels = (-8.0, -4.0, 4.0, 8.0, 12.0)
    for row, (name, cbar_label, cmap, vmin, vmax) in enumerate(variables):
        images = []
        for col, (index, step, label) in enumerate(zip(indices, steps, labels)):
            values = np.asarray(data[name][index])
            if name in ("specific_cloud_g_kg-1", "specific_rain_ug_kg-1"):
                values = np.maximum(values, 0.0)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0.0 else None
            image = axes[row, col].imshow(
                values, origin="lower", extent=extent, cmap=cmap, vmin=None if norm else vmin,
                vmax=None if norm else vmax, norm=norm,
            )
            images.append(image)
            axes[row, col].contour(
                np.asarray(data["x_km"]), np.asarray(data["y_km"]),
                data["relative_vorticity_1e5_s-1"][index], levels=vort_levels,
                colors="k", linewidths=0.22, alpha=0.3,
            )
            if row == 0:
                axes[row, col].set_title(
                    f"{label}\n$t={step * 100 / 3600:.2f}$ h",
                    fontsize=12.2,
                    linespacing=1.12,
                )
        colorbar = fig.colorbar(
            images[-1], ax=axes[row, :], shrink=0.88, pad=0.012, aspect=24
        )
        colorbar.set_label(cbar_label, fontsize=12.5)
        colorbar.ax.tick_params(labelsize=10)
    for row in range(4):
        axes[row, 0].set_ylabel("y (km)", fontsize=11.5)
    for col in range(5):
        axes[-1, col].set_xlabel("x (km)", fontsize=11.5)
    for axis in axes.reshape(-1):
        axis.set_aspect("equal")
        axis.tick_params(labelsize=9.5)
    panel_labels(axes, fontsize=12)
    paths = save_figure(fig, output / "figure3_test2b_event_gallery")
    return paths, {
        "state_indices": list(steps),
        "times_s": [float(step * 100) for step in steps],
        "variables_units": {name: label for name, label, *_ in variables},
        "color_limits": {name: [vmin, vmax] for name, _, _, vmin, vmax in variables},
        "overlay": {"variable": "relative_vorticity_1e5_s-1", "levels": list(vort_levels)},
        "display_clipping": "negative numerical Qc/Qr undershoots clipped to zero in nonnegative water panels; raw values retained in cache",
    }


def figure4(a, b, output: Path):
    ta = a["time_s"] / 3600.0
    tb = b["time_s"] / 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.1), constrained_layout=True, sharex=True)
    axes[0, 0].plot(ta, a["Qc_mass"] / 1e12, color=COLORS["test2a"], lw=1.7, label="Test 2A")
    axes[0, 0].plot(tb, b["Qc_mass"] / 1e12, color=COLORS["test2b"], lw=1.7, label="Test 2B")
    axes[0, 0].set_ylabel(r"$M_c$ ($10^{12}$ m$^3$)")
    axes[0, 0].set_title("Integrated cloud water")
    axes[0, 0].legend()

    axes[0, 1].plot(ta, 1000 * a["specific_Qc_maximum"], color=COLORS["test2a"], lw=1.7)
    axes[0, 1].plot(tb, 1000 * b["specific_Qc_maximum"], color=COLORS["test2b"], lw=1.7)
    axes[0, 1].axhline(0.1, color="0.25", lw=1.0, ls=":")
    axes[0, 1].set_ylabel(r"max $q_c$ (g kg$^{-1}$)")
    axes[0, 1].set_title("Rain-threshold separation")

    axes[1, 0].plot(ta, a["rain_water_mass"] / 1e8, color=COLORS["test2a"], lw=1.7)
    axes[1, 0].plot(tb, b["rain_water_mass"] / 1e8, color=COLORS["test2b"], lw=1.7)
    axes[1, 0].set_ylabel(r"$M_r$ ($10^8$ m$^3$)")
    axes[1, 0].set_title("Integrated rain water")

    for data, time, color, label in ((a, ta, COLORS["test2a"], "Test 2A"), (b, tb, COLORS["test2b"], "Test 2B")):
        lower = 100.0 * (data["saturation_ratio_minimum"] - 1.0)
        upper = 100.0 * (data["saturation_ratio_maximum"] - 1.0)
        axes[1, 1].fill_between(time, lower, upper, color=color, alpha=0.18)
        axes[1, 1].plot(time, lower, color=color, lw=0.8)
        axes[1, 1].plot(time, upper, color=color, lw=0.8, label=label)
    axes[1, 1].axhline(0, color="0.3", lw=0.7)
    axes[1, 1].set_ylabel(r"$100(q_v/q_{sat}-1)$ (%)")
    axes[1, 1].set_title("Post-prefix saturation envelope")
    axes[1, 1].legend(loc="lower left")

    for ax in axes[1, :]:
        ax.set_xlabel("time (h)")
    for ax in axes.reshape(-1):
        ax.grid(color="0.9", lw=0.6)
        ax.axvline(5100 / 3600, color=COLORS["onset"], lw=0.8, ls="--", alpha=0.8)
    panel_labels(axes)
    paths = save_figure(fig, output / "figure4_test2a_test2b_comparison")
    return paths, {
        "state_indices": [0, 160],
        "times_s": [0.0, 16000.0],
        "variables_units": {"Qc_mass": "m3", "max_qc": "g kg-1", "Qr_mass": "m3", "saturation_departure": "%"},
        "interpretation_limit": "Test 2A and Test 2B differ in both initial vapor loading and mesh resolution; regime comparison only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test2a-cache", required=True, type=Path)
    parser.add_argument("--test2b-cache", required=True, type=Path)
    parser.add_argument("--test2a-summary", required=True, type=Path)
    parser.add_argument("--test2b-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    style()
    script = Path(__file__).resolve()
    summary_a = load_json(args.test2a_summary)
    summary_b = load_json(args.test2b_summary)
    with np.load(args.test2a_cache) as loaded:
        data_a = {name: loaded[name] for name in loaded.files}
    with np.load(args.test2b_cache) as loaded:
        data_b = {name: loaded[name] for name in loaded.files}

    results = []
    results.append(("figure1", *figure1(args.output)))
    results.append(("figure2", *figure2(data_a, data_b, args.output)))
    results.append(("figure3", *figure3(data_b, args.output)))
    for figure_id, paths, detail in results:
        sidecar = {
            "format_version": 1,
            "figure_id": figure_id,
            "outputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
            "script": str(script),
            "script_sha256": sha256(script),
            "source_truth_paths": [summary_a["source_truth"], summary_b["source_truth"]],
            "source_caches": [
                {"path": str(args.test2a_cache.resolve()), "sha256": sha256(args.test2a_cache)},
                {"path": str(args.test2b_cache.resolve()), "sha256": sha256(args.test2b_cache)},
            ],
            "coordinate_units": "km",
            "deterministic_rendering": {"backend": "Agg", "font": "DejaVu Sans", "png_dpi": 300, "fixed_pdf_date": "2026-08-29T00:00:00Z"},
            **detail,
        }
        write_json(args.output / f"{figure_id}.json", sidecar)
    print("generated", len(results), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
