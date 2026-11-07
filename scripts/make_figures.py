"""Figures and tables from results/study.npz, in English (paper/figures, paper/tables) or in French
(``make_figures.py fr``: paper/figures/fr, paper/tables/fr)."""

from __future__ import annotations

import locale
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inroom import analysis as an  # noqa: E402
from inroom import crossover, cta2034, ensemble, sphere  # noqa: E402
from inroom import plotstyle as ps  # noqa: E402
from inroom import room as rm  # noqa: E402
from inroom.signal import db10, log_grid, smooth_power  # noqa: E402
from inroom.speaker import TWO_WAY, TWO_WAY_WG, FarField  # noqa: E402
from inroom.study import power  # noqa: E402

LANG = "en"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
KIND_COLORS = {"lr2": ps.SLOTS[3], "lr4": ps.SLOTS[0], "lr8": ps.SLOTS[6], "fir": ps.SLOTS[1]}
SURF_COLORS = {"floor": ps.SLOTS[1], "ceiling": ps.SLOTS[3], "left": ps.SLOTS[0], "right": ps.SLOTS[6],
               "listener_wall": ps.SLOTS[2], "speaker_wall": ps.SLOTS[4]}
W1, W2 = 3.4, 7.0  # figure widths (in): one column, two columns
NBSP, NNBSP = " ", " "


def L(en: str, fr: str) -> str:
    """The text in the language of the run."""
    return fr if LANG == "fr" else en


def num(x, fmt: str = ".2f") -> str:
    """A number, with a decimal comma in French."""
    s = format(float(x), fmt)
    return s.replace(".", ",") if LANG == "fr" else s


def kind_label(kind: str) -> str:
    return {"lr2": "LR2", "lr4": "LR4", "lr8": "LR8", "fir": L("brickwall FIR", "FIR brickwall")}[kind]


def surf_label(surf: str) -> str:
    return {"left": L("near side wall", "mur latéral proche"), "right": L("far side wall", "mur latéral éloigné"),
            "listener_wall": L("wall behind listener", "mur derrière l’auditeur"),
            "speaker_wall": L("wall behind loudspeaker", "mur derrière l’enceinte"),
            "floor": L("floor", "sol"), "ceiling": L("ceiling", "plafond")}[surf]


def layout_label(layout: str) -> str:
    return {"two_way": L("dome", "dôme"), "two_way_wg": L("dome in a waveguide", "dôme en guide d’onde")}[layout]


def colon() -> str:
    """A colon, with the no-break space French typography puts before it."""
    return NBSP + ":" if LANG == "fr" else ":"


def percent() -> str:
    return NNBSP + "%" if LANG == "fr" else " %"


def name(layout="two_way", orient="upright", kind="lr4"):
    return f"{layout}-{orient}-{kind}"


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    plt.close(fig)


def smooth(f, p, frac=12):
    return smooth_power(f, p, frac)


def level_axis(ax, label=None):
    ax.set_ylabel(label or L("level (dB)", "niveau (dB)"))


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def fig_setup():
    """Plan and elevation of the living room, with the first-order reflection paths."""
    st = ensemble.named()["living"]
    lx, ly, lz = st.room.dims
    s, r = st.source, st.receiver
    im = rm.image_sources(st, 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[lx, ly * 0.9]))
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.grid(False, which="both")
    a.add_patch(plt.Rectangle((0, 0), lx, ly, fill=False, ec=ps.INK_2, lw=1.0))
    for p in st.room.patches:
        if p.surface == "floor":
            u0, u1, v0, v1 = p.rect
            a.add_patch(plt.Rectangle((u0, v0), u1 - u0, v1 - v0, color=ps.NEUTRAL, ec=ps.AXIS, lw=0.5))
            a.text(u1 - 0.05, v0 + 0.08, L("rug", "tapis"), ha="right", fontsize=6.5, color=ps.MUTED)
    mirror = np.array([lx - s[0], s[1], s[2]])
    for i in np.flatnonzero(im["order"] == 1):
        surf = rm.SURFACES[im["first"][i]]
        p = im["point"][i]
        col = SURF_COLORS[surf]
        if surf in ("floor", "ceiling"):
            b.plot([ly - s[1], ly - p[1], ly - r[1]], [s[2], p[2], r[2]], color=col, lw=0.9)
        else:
            a.plot([s[0], p[0], r[0]], [s[1], p[1], r[1]], color=col, lw=0.9, label=surf_label(surf))
    a.plot([s[0], r[0]], [s[1], r[1]], color=ps.INK, lw=1.1)
    b.plot([ly - s[1], ly - r[1]], [s[2], r[2]], color=ps.INK, lw=1.1)
    for x, y in ((s[0], s[1]), (mirror[0], mirror[1])):
        a.add_patch(plt.Rectangle((x - 0.1, y), 0.2, 0.28, color=ps.INK_2))
    axis = st.axes[2]
    a.annotate("", xy=(s[0] + 0.6 * axis[0], s[1] + 0.6 * axis[1]), xytext=(s[0], s[1]),
               arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=0.6))
    a.add_patch(plt.Circle((r[0], r[1]), 0.12, color=ps.INK))
    a.set_xlim(-0.15, lx + 0.15)
    a.set_ylim(-0.15, ly + 0.15)
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title(L("plan", "vue en plan"))
    b.add_patch(plt.Rectangle((0, 0), ly, lz, fill=False, ec=ps.INK_2, lw=1.0))
    b.add_patch(plt.Rectangle((ly - s[1] - 0.28, s[2] - 0.29), 0.28, 0.36, color=ps.INK_2))
    b.add_patch(plt.Circle((ly - r[1], r[2]), 0.12, color=ps.INK))
    b.text(0.5 * (ly - s[1] + ly - r[1]), 0.12, surf_label("floor"), color=SURF_COLORS["floor"], ha="center", fontsize=7)
    b.text(0.5 * (ly - s[1] + ly - r[1]), lz - 0.25, surf_label("ceiling"), color=SURF_COLORS["ceiling"],
           ha="center", fontsize=7)
    b.set_xlim(-0.15, ly + 0.15)
    b.set_ylim(-0.15, lz + 0.15)
    b.set_xlabel(L("distance from the wall behind the loudspeakers (m)", "distance au mur derrière les enceintes (m)"))
    b.set_ylabel("z (m)")
    b.set_title(L("elevation", "coupe"))
    a.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
    fig.tight_layout()
    save(fig, "setup")


def fig_kernel_maps(study: an.Study):
    """Where the energy leaves the loudspeaker: CTA-2034 PIR weights against the rooms."""
    d = study.raw
    n = study.n_random
    f = study.freqs
    j = int(np.argmin(np.abs(d["map_freqs"] - 2200.0)))
    fj = int(np.argmin(np.abs(f - 2200.0)))
    fig, (a, b) = plt.subplots(2, 1, figsize=(W2, 6.4))
    for ax in (a, b):
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_xlabel(L("azimuth (deg, + toward the listener's right)", "azimut (°, + vers la droite de l’auditeur)"))
        ax.set_ylabel(L("elevation (deg)", "élévation (°)"))
        ax.set_aspect("equal")
    parts = {"lw": cta2034.kernel("lw") * 0.12, "er": cta2034.kernel("er") * 0.44, "sp": cta2034.kernel("sp") * 0.44}
    az, el = sphere.to_angles(cta2034.DIRECTIONS)
    handles = []
    for label, color, part in ((f"SP (44{percent()})", ps.SLOTS[2], parts["sp"]),
                               (f"ER (44{percent()})", ps.SLOTS[1], parts["er"]),
                               (f"LW (12{percent()})", ps.SLOTS[0], parts["lw"])):
        m = part > 0
        a.scatter(az[m], el[m], s=part[m] * 2500, color=color, alpha=0.55, lw=0)
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=6, color=color, alpha=0.7, label=label))
    a.scatter(az, el, s=2, color=ps.INK, lw=0)
    a.set_title(L("CTA-2034 Predicted In-Room Response: weight of each of the 70 directions (area)",
                  f"PIR de la CTA-2034{colon()} poids de chacune des 70 directions (surface)"))
    a.legend(handles=handles, loc="lower left", fontsize=7, ncol=3, framealpha=0.9, frameon=True)

    grid = d["grid"]
    dens = d["diffuse_mean"][:, fj]
    gaz, gel = sphere.to_angles(grid)
    sc = b.scatter(gaz, gel, c=dens / np.mean(dens), s=9, cmap="Greys", vmin=0.3, vmax=1.4, lw=0)
    cb = fig.colorbar(sc, ax=b, shrink=0.6, pad=0.01)
    cb.set_label(L("diffuse field: share per direction\n(relative to the mean)",
                   f"champ diffus{colon()} part par direction\n(relative à la moyenne)"), fontsize=7)
    starts = d["image_starts"]
    cat = d["image_cat"][: starts[n]]
    dirs = d["image_dirs"][: starts[n]]
    iaz, iel = sphere.to_angles(dirs)
    total = study.omni[:n].sum(axis=1)[:, fj]
    room_of = np.repeat(np.arange(n), np.diff(starts[: n + 1]))
    share = d["image_weights"][: starts[n], j] / total[room_of]
    m = cat == 0
    b.scatter(iaz[m], iel[m], s=share[m] * 600, color=ps.INK, alpha=0.35, lw=0)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=ps.INK, alpha=0.6,
                          label=L("direct sound", "son direct"))]
    for c, surf in enumerate(rm.SURFACES):
        m = cat == 1 + c
        b.scatter(iaz[m], iel[m], s=share[m] * 600, color=SURF_COLORS[surf], alpha=0.4, lw=0)
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=5, color=SURF_COLORS[surf], alpha=0.7,
                                  label=surf_label(surf)))
    b.set_title(L("300 rooms at 2.2 kHz: direct sound and first reflections (area: share of the energy)",
                  f"300 pièces à 2,2{NBSP}kHz{colon()} son direct et premières réflexions "
                  f"(surface{colon()} part de l’énergie)"))
    b.legend(handles=handles, loc="lower left", fontsize=6.5, ncol=4, framealpha=0.9, frameon=True)
    fig.tight_layout()
    save(fig, "kernel_maps")


def fig_shares(study: an.Study):
    """Shares of the energy at the listener (omnidirectional source) against frequency."""
    f = study.freqs
    sh = an.shares(study)
    groups = [(L("direct sound", "son direct"), [0], ps.INK),
              (L("first order, floor + ceiling", "1er ordre, sol + plafond"), [5, 6], ps.SLOTS[1]),
              (L("first order, walls", "1er ordre, murs"), [1, 2, 3, 4], ps.SLOTS[0]),
              (L("second and third order", "2e et 3e ordres"), [7, 8], ps.SLOTS[6]),
              (L("diffuse field", "champ diffus"), [9], ps.SLOTS[2])]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bottom = np.zeros(len(f))
    for label, idx, color in groups:
        val = np.median(sh[:, idx].sum(axis=1), axis=0)
        ax.fill_between(f, bottom * 100, (bottom + val) * 100, color=color, alpha=0.55, lw=0, label=label)
        bottom = bottom + val
    ps.freq_axis(ax, 150, 20000)
    ax.set_ylim(0, 100)
    ax.set_ylabel(L("share of the energy (%)", "part de l’énergie (%)"))
    ax.set_title(L("energy at the listener, median of 300 rooms", "énergie reçue par l’auditeur, médiane de 300 pièces"))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.tight_layout()
    save(fig, "shares")


def fig_reflection_angles(study: an.Study):
    """Directions in which the first reflections leave the loudspeaker, against CTA-2034's angles."""
    d = study.raw
    n = study.n_random
    starts = d["image_starts"]
    cat = d["image_cat"][: starts[n]]
    az, el = sphere.to_angles(d["image_dirs"][: starts[n]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.4))
    bins = np.arange(-90, 91, 2.5)
    for c, surf in ((5, "floor"), (6, "ceiling")):
        a.hist(el[cat == c], bins=bins, color=SURF_COLORS[surf], alpha=0.7, label=surf_label(surf))
    for ang in cta2034.FLOOR:
        a.axvline(ang, color=SURF_COLORS["floor"], lw=0.8, ls="--")
    for ang in cta2034.CEILING:
        a.axvline(ang, color=SURF_COLORS["ceiling"], lw=0.8, ls="--")
    a.set_xlim(-70, 70)
    a.set_xlabel(L("elevation (deg)", "élévation (°)"))
    a.set_ylabel(L("rooms", "pièces"))
    a.set_title(L("floor and ceiling (dashed: CTA-2034 angles)", f"sol et plafond (tirets{colon()} angles de la CTA-2034)"))
    a.legend(fontsize=6.5)
    bins = np.arange(-180, 181, 5)
    for c, surf in ((1, "left"), (2, "right"), (3, "listener_wall"), (4, "speaker_wall")):
        b.hist(az[cat == c], bins=bins, color=SURF_COLORS[surf], alpha=0.7, label=surf_label(surf))
    for ang in cta2034.SIDE_WALL:
        b.axvline(ang, color=ps.SLOTS[0], lw=0.6, ls="--")
    for ang in cta2034.REAR_WALL + (-180,):
        b.axvline(ang, color=SURF_COLORS["speaker_wall"], lw=0.8, ls=":")
    for ang in cta2034.FRONT_WALL:
        b.axvline(ang, color=SURF_COLORS["listener_wall"], lw=0.6, ls="--")
    b.set_xlim(-180, 180)
    b.set_xticks(range(-180, 181, 45))
    b.set_xlabel(L("azimuth (deg)", "azimut (°)"))
    b.set_title(L("walls (dashed: side and front walls, dotted: rear wall)",
                  f"murs (tirets{colon()} latéraux et avant, pointillés{colon()} arrière)"))
    b.legend(fontsize=6, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "reflection_angles")


def weights_label(w) -> str:
    return f"{num(w[0])} LW + {num(w[1])} ER + {num(w[2])} SP ({L('sphere', 'sphère')})"


def fig_spin(study: an.Study, layout="two_way", stem="spin_dome"):
    """The CTA-2034 curves of the upright two-way (LR4), and its in-room responses."""
    f = study.freqs
    v = name(layout, "upright", "lr4")
    c = study.curves(v)
    w = an.fit_weights(study)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    for key, label, color, ls in (("on_axis", L("on axis", "dans l’axe"), ps.INK, "-"),
                                  ("lw", L("listening window", "fenêtre d’écoute"), ps.SLOTS[0], "-"),
                                  ("er", L("early reflections", "premières réflexions"), ps.SLOTS[1], "-"),
                                  ("sp", L("sound power, two orbits", "puissance, deux orbites"), ps.SLOTS[2], "--"),
                                  ("sp_sphere", L("sound power, sphere", "puissance, sphère"), ps.SLOTS[2], "-")):
        a.semilogx(f, db10(smooth(f, c[key])), color=color, ls=ls, label=label)
    a.set_title(L("CTA-2034 curves (far field)", "courbes CTA-2034 (champ lointain)"))
    rooms = study.rooms(v, "steady", study.random())
    lo, med, hi = np.percentile(db10(rooms), [10, 50, 90], axis=0)
    b.fill_between(f, lo, hi, color=ps.SLOTS[2], alpha=0.2, lw=0, label=L("rooms, 10-90 %", f"pièces, 10–90{percent()}"))
    b.semilogx(f, med, color=ps.SLOTS[2], label=L("rooms, median", "pièces, médiane"))
    early = study.rooms(v, "early", study.random())
    b.semilogx(f, np.median(db10(early), axis=0), color=ps.SLOTS[1],
               label=L("rooms, direct + first reflections", "pièces, direct + premières réflexions"))
    b.semilogx(f, db10(smooth(f, c["pir"])), color=ps.INK, ls="--", label=L("CTA-2034 PIR", "PIR CTA-2034"))
    b.semilogx(f, db10(smooth(f, an.pir(c, w, "sp_sphere"))), color=ps.INK, ls=":", label=weights_label(w))
    b.set_title(L("in room, relative to an omnidirectional source", "dans la pièce, relatif à une source omnidirectionnelle"))
    for ax in (a, b):
        ps.freq_axis(ax, 150, 20000)
        ax.set_ylim(-14, 3)
        ax.legend(fontsize=6.3, loc="lower left")
    level_axis(a)
    fig.tight_layout()
    save(fig, stem)


def fig_sp_orbits(study: an.Study):
    """Sound power from two orbits against the sphere, for each crossover."""
    f = study.freqs
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    for ax, layout in zip(axes, ("two_way", "two_way_wg")):
        for kind in crossover.KINDS:
            c = study.curves(name(layout, "upright", kind))
            ax.semilogx(f, db10(c["sp"]) - db10(c["sp_sphere"]), color=KIND_COLORS[kind], label=kind_label(kind))
        ps.freq_axis(ax, 300, 20000)
        ax.set_title(f"{L('two-way', 'deux voies')}, {layout_label(layout)}")
        ax.axhline(0, color=ps.AXIS, lw=0.6)
    axes[0].set_ylabel(L("SP two orbits - SP sphere (dB)", "SP deux orbites − SP sphère (dB)"))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "sp_orbits")


def fig_errors(study: an.Study):
    """Mean error of the predictions against the rooms (test half), every variant."""
    f = study.freqs
    w = an.fit_weights(study)
    preds = [(L("CTA-2034 PIR", "PIR CTA-2034"), lambda c: an.pir(c)),
             (L("same weights, sound power of the sphere", "mêmes poids, puissance sur la sphère"),
              lambda c: an.pir(c, sp="sp_sphere")),
             (weights_label(w), lambda c: an.pir(c, w, "sp_sphere"))]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
    for ax, (title, fn) in zip(axes, preds):
        spread = []
        for v in study.variants:
            c = study.curves(v)
            _, bias = an.errors(study, v, fn(c))
            ax.semilogx(f, bias, color=ps.SLOTS[0] if "two_way-" in v else ps.SLOTS[1], lw=0.6, alpha=0.7)
            e = db10(fn(c))[None, :] - db10(study.rooms(v, "steady", study.test()))
            spread.append(np.percentile(e, [10, 90], axis=0) - e.mean(axis=0))
        lo, hi = np.mean(spread, axis=0)
        ax.fill_between(f, lo, hi, color=ps.NEUTRAL, lw=0, zorder=0,
                        label=L("room to room, 10-90 %", f"d’une pièce à l’autre, 10–90{percent()}"))
        ax.axhline(0, color=ps.AXIS, lw=0.6)
        ps.freq_axis(ax, 300, 16000, sparse=True)
        ax.set_title(title, fontsize=7)
    axes[0].set_ylabel(L("prediction - room (dB)", "prédiction − pièce (dB)"))
    axes[0].plot([], [], color=ps.SLOTS[0], label=L("dome", "dôme"))
    axes[0].plot([], [], color=ps.SLOTS[1], label=L("waveguide", "guide d’onde"))
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "errors")


def fig_rotation(study: an.Study):
    """The same loudspeaker upright and on its side (LR4)."""
    f = study.freqs
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
    for orient, ls in (("upright", "-"), ("side", "--")):
        v = name("two_way", orient, "lr4")
        c = study.curves(v)
        lab = L("upright", "debout") if orient == "upright" else L("on its side", "couchée")
        axes[0].semilogx(f, db10(smooth(f, c["lw"])), color=ps.SLOTS[0], ls=ls, label=f"LW, {lab}")
        axes[0].semilogx(f, db10(smooth(f, c["er"])), color=ps.SLOTS[1], ls=ls, label=f"ER, {lab}")
        axes[0].semilogx(f, db10(smooth(f, c["pir"])), color=ps.INK, ls=ls, label=f"PIR, {lab}")
        for ax, which in ((axes[1], "early"), (axes[2], "steady")):
            r = db10(study.rooms(v, which, study.random()))
            ax.semilogx(f, np.median(r, axis=0), color=ps.INK, ls=ls, label=f"{L('median', 'médiane')}, {lab}")
            ax.semilogx(f, np.percentile(r, 10, axis=0), color=ps.SLOTS[1], ls=ls, label=f"10{percent()}, {lab}")
    axes[0].set_title("CTA-2034")
    axes[1].set_title(L("rooms: direct + first reflections", f"pièces{colon()} direct + premières réflexions"))
    axes[2].set_title(L("rooms: steady state", f"pièces{colon()} régime établi"))
    for ax in axes:
        ps.freq_axis(ax, 300, 20000, sparse=True)
        ax.set_ylim(-14, 2)
        ax.legend(fontsize=5.6, loc="lower left")
    level_axis(axes[0])
    fig.tight_layout()
    save(fig, "rotation")


def fig_crossovers(study: an.Study):
    """In-room responses of the four crossovers, dome and waveguide."""
    f = study.freqs
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.6), sharex=True)
    for col, layout in enumerate(("two_way", "two_way_wg")):
        title = L("dome", "dôme") if layout == "two_way" else L("waveguide", "guide d’onde")
        for kind in crossover.KINDS:
            v = name(layout, "upright", kind)
            c = study.curves(v)
            col_ = KIND_COLORS[kind]
            r = np.median(db10(study.rooms(v, "steady", study.random())), axis=0)
            axes[0, col].semilogx(f, r, color=col_, label=kind_label(kind))
            axes[0, col].semilogx(f, db10(c["pir"]), color=col_, ls=":", lw=0.9)
            e = np.median(db10(study.rooms(v, "early", study.random())), axis=0)
            axes[1, col].semilogx(f, e, color=col_, label=kind_label(kind))
        axes[0, col].set_title(L(f"{title}: steady state (dotted: CTA-2034 PIR)",
                                 f"{title}{colon()} régime établi (pointillés{colon()} PIR CTA-2034)"), fontsize=7.5)
        axes[1, col].set_title(L(f"{title}: direct + first reflections", f"{title}{colon()} direct + premières réflexions"),
                               fontsize=7.5)
        for ax in axes[:, col]:
            ps.freq_axis(ax, 500, 10000)
            ax.set_xticks([500, 1000, 2000, 5000, 10000])
            ax.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
    for ax in axes[:, 0]:
        level_axis(ax, L("level, median of 300 rooms (dB)", "niveau, médiane de 300 pièces (dB)"))
    for ax in axes[0]:
        ax.set_ylim(-11, -1)
    for ax in axes[1]:
        ax.set_ylim(-5, 1)
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "crossovers")


def path_spectra():
    """Responses of the living room's direct sound, floor and ceiling reflections, for each
    layout and crossover: ``{layout: (freqs, {kind: {path: power}})}``."""
    f = log_grid(300.0, 20000.0, 96)
    st = ensemble.named()["living"]
    im = rm.image_sources(st, 1)
    paths = {"direct": im["direction"][0]}
    for i in np.flatnonzero(im["order"] == 1):
        surf = rm.SURFACES[im["first"][i]]
        if surf in ("floor", "ceiling"):
            paths[surf] = im["direction"][i]
    out = {}
    for layout in (TWO_WAY, TWO_WAY_WG):
        far = FarField(layout, f)
        on = {k: v[0] for k, v in far.responses(np.array([[0.0, 0.0, 1.0]])).items()}
        resp = far.responses(np.array(list(paths.values())))
        res = {}
        for kind in crossover.KINDS:
            p = power(crossover.filters(kind, f, layout, on), resp)
            res[kind] = {k: p[i] for i, k in enumerate(paths)}
        out[layout.name] = (f, res, paths)
    return out


def notch_width(f, p, depth=6.0):
    """Width (Hz) of the band around the deepest dip, within an octave of the crossover, where the
    response stays ``depth`` dB below the straight line fitted from 600 Hz to 8 kHz."""
    m = an.band(f, an.ANOMALY_BAND)
    r, _ = an.line_residual(f, p)
    fb = f[m]
    sel = (fb >= an.CROSSOVER_BAND[0]) & (fb <= an.CROSSOVER_BAND[1])
    i = int(np.flatnonzero(sel)[np.argmin(r[sel])])
    lo = i
    while lo > 0 and r[lo - 1] < -depth:
        lo -= 1
    hi = i
    while hi < len(r) - 1 and r[hi + 1] < -depth:
        hi += 1
    return (fb[hi] - fb[lo]) if r[i] < -depth else 0.0, fb[i], -r[i]


def fig_paths(study: an.Study, spectra, numbers=None):
    """What each reflection carries, against the room's average: the living room."""
    room_index = study.n_random + study.named.index("living")
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.6), sharex=True)
    for row, layout in enumerate(("two_way", "two_way_wg")):
        f, res, paths = spectra[layout]
        ax = axes[row, 0]
        for kind in ("lr2", "lr4", "fir"):
            for path, ls in (("floor", "-"), ("ceiling", "--")):
                p = res[kind][path]
                ax.semilogx(f, db10(p), color=KIND_COLORS[kind], ls=ls, lw=0.9,
                            label=f"{kind_label(kind)}, {surf_label(path)}")
        el = {k: f"{sphere.to_angles(v)[1]:.0f}".replace("-", "\u2212") for k, v in paths.items()}
        ax.set_title(L(f"{layout_label(layout)}: floor ({el['floor']}°) and ceiling ({el['ceiling']}°) bounces",
                       f"{layout_label(layout)}{colon()} rebonds sol ({el['floor']}°) et plafond ({el['ceiling']}°)"),
                     fontsize=8)
        ax.set_ylim(-30, 3)
        ax2 = axes[row, 1]
        fs = study.freqs
        for kind in ("lr2", "lr4", "fir"):
            v = name(layout, "upright", kind)
            e = study.data[v]["rooms"][room_index].sum(axis=0) / study.omni[room_index].sum(axis=0)
            ax2.semilogx(fs, db10(e), color=KIND_COLORS[kind], label=kind_label(kind))
        ax2.set_title(L(f"{layout_label(layout)}: the same room, steady state",
                        f"{layout_label(layout)}{colon()} la même pièce, régime établi"))
        ax2.set_ylim(-12, 0)
        for a in (ax, ax2):
            ps.freq_axis(a, 500, 10000)
            a.set_xticks([500, 1000, 2000, 5000, 10000])
            a.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
        level_axis(ax, L("level re on axis (dB)", "niveau relatif à l’axe (dB)"))
        level_axis(ax2, L("level re omnidirectional (dB)", "niveau relatif à l’omni (dB)"))
        if numbers is not None:
            for kind in ("lr2", "lr4", "fir"):
                for path in ("floor", "ceiling"):
                    width, fc, depth = notch_width(f, res[kind][path])
                    tag = {"two_way": "Dome", "two_way_wg": "Wg"}[layout]
                    word = {"lr2": "LrTwo", "lr4": "LrFour", "fir": "Fir"}[kind]
                    numbers[f"notch{tag}{word}{path.title()}Width"] = num(width, ".0f")
                    numbers[f"notch{tag}{word}{path.title()}Depth"] = num(depth, ".0f")
    axes[0, 0].legend(fontsize=5.8, loc="lower left", ncol=2)
    axes[0, 1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "paths")


def fig_vertical_colouration(study: an.Study):
    """Each direction's own deviation within an octave of the crossover, along the vertical
    orbit, with the directions of the rooms' floor and ceiling bounces."""
    d = study.raw
    n = study.n_random
    f = log_grid(300.0, 20000.0, 96)
    angles = np.arange(-90.0, 90.1, 1.0)
    dirs = np.stack([np.zeros_like(angles), np.sin(np.radians(angles)), np.cos(np.radians(angles))], axis=1)
    starts = d["image_starts"]
    cat = d["image_cat"][: starts[n]]
    _, el = sphere.to_angles(d["image_dirs"][: starts[n]])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for ax, layout in zip(axes, (TWO_WAY, TWO_WAY_WG)):
        for c, surf in ((5, "floor"), (6, "ceiling")):
            lo, hi = np.percentile(el[cat == c], [10, 90])
            ax.axvspan(lo, hi, color=SURF_COLORS[surf], alpha=0.15, lw=0)
        far = FarField(layout, f)
        on = {k: v[0] for k, v in far.responses(np.array([[0.0, 0.0, 1.0]])).items()}
        resp = far.responses(dirs)
        for kind in crossover.KINDS:
            p = power(crossover.filters(kind, f, layout, on), resp)
            ax.plot(angles, an.local_anomaly(f, p), color=KIND_COLORS[kind], label=kind_label(kind))
        ax.set_xlim(-90, 90)
        ax.set_xticks(range(-90, 91, 30))
        ax.set_xlabel(L("vertical angle (deg)", "angle vertical (°)"))
        ax.set_title(f"{L('two-way', 'deux voies')}, {layout_label(layout.name)}")
    axes[0].set_ylabel(L("deviation, 1.1-4.4 kHz (dB RMS)", f"écart, 1,1–4,4{NBSP}kHz (dB RMS)"))
    axes[0].legend(fontsize=6.5, loc="upper center")
    fig.tight_layout()
    save(fig, "vertical_colouration")


COLOURATION_METRICS = ("pir", "fit", "steady", "reflections", "vertical")


def colouration_values(study: an.Study, v: str, w) -> list:
    return [an.cta_path_anomaly(study, v), an.cta_path_anomaly(study, v, w, "sp_sphere"),
            float(np.median(an.path_anomaly(study, v, "steady"))),
            float(np.median(an.path_anomaly(study, v, "reflections"))),
            float(np.median(an.path_anomaly(study, v, "vertical")))]


def fig_colouration(study: an.Study):
    """Colouration of the paths: each direction's own deviation within an octave of the
    crossover, weighted by the energy it brings, for every crossover."""
    w = an.fit_weights(study)
    labels = [L("CTA-2034\nPIR weights", "poids de la\nPIR CTA-2034"), L("updated\nweights", "poids\nmis à jour"),
              L("rooms,\nsteady state", "pièces,\nrégime\nétabli"),
              L("rooms, first\nreflections", "pièces,\npremières\nréflexions"),
              L("rooms, floor\n+ ceiling", "pièces,\nsol +\nplafond")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    width = 0.2
    for ax, layout in zip(axes, ("two_way", "two_way_wg")):
        for i, kind in enumerate(crossover.KINDS):
            vals = colouration_values(study, name(layout, "upright", kind), w)
            ax.bar(np.arange(len(labels)) + (i - 1.5) * width, vals, width, color=KIND_COLORS[kind],
                   label=kind_label(kind))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.set_title(f"{L('two-way', 'deux voies')}, {layout_label(layout)}")
        ax.grid(False, axis="x", which="both")
    axes[0].set_ylabel(L("colouration of the paths,\n1.1-4.4 kHz (dB RMS)",
                         f"coloration des chemins,\n1,1–4,4{NBSP}kHz (dB RMS)"))
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "colouration")


def fig_polar_maps():
    """Vertical-plane directivity of the four crossovers, dome and waveguide (normalized to the
    on-axis response)."""
    f = log_grid(300.0, 20000.0, 48)
    angles = np.arange(-90, 90.1, 1.0)
    dirs = np.stack([np.zeros_like(angles), np.sin(np.radians(angles)), np.cos(np.radians(angles))], axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.8), sharex=True, sharey=True)
    for row, layout in enumerate((TWO_WAY, TWO_WAY_WG)):
        far = FarField(layout, f)
        resp = far.responses(dirs)
        on = {k: v[np.argmin(np.abs(angles))] for k, v in resp.items()}
        for col, kind in enumerate(crossover.KINDS):
            x = crossover.filters(kind, f, layout, on)
            p = power(x, resp)
            lev = db10(p) - db10(p[np.argmin(np.abs(angles))])[None, :]
            ax = axes[row, col]
            im = ax.pcolormesh(f, angles, lev, cmap=ps.DIVERGING, vmin=-18, vmax=18, shading="auto", rasterized=True)
            ax.set_xscale("log")
            ax.set_xticks([1000, 10000])
            ax.set_xticklabels(["1k", "10k"])
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.grid(False, which="both")
            if row == 0:
                ax.set_title(kind_label(kind))
        axes[row, 0].set_ylabel((L("dome", "dôme") if row == 0 else L("waveguide", "guide d’onde"))
                                + "\n" + L("vertical angle (deg)", "angle vertical (°)"))
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label(L("level re on axis (dB)", "niveau relatif à l’axe (dB)"))
    save(fig, "polar_maps")


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

def tex_write(name, text):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(text)


def tables(study: an.Study, numbers: dict):
    f = study.freqs
    n = study.n_random
    meta = study.meta
    to = L("to", "à")

    # Ensemble.
    dims = meta["dims"][:n]
    rt = meta["rt60"][:n]
    rows = [(L("width (m)", "largeur (m)"), dims[:, 0]), (L("length (m)", "longueur (m)"), dims[:, 1]),
            (L("height (m)", "hauteur (m)"), dims[:, 2]), ("volume (m$^3$)", np.prod(dims, axis=1)),
            (L("listening distance (m)", "distance d’écoute (m)"), meta["listen_distance"][:n].astype(float)),
            (L("loudspeaker spacing (m)", "écartement des enceintes (m)"), meta["spacing"][:n].astype(float)),
            (L("baffle to wall behind (m)", "distance du baffle au mur (m)"), meta["wall_distance"][:n].astype(float)),
            (L("$T_{60}$ at 500 Hz (s)", "$T_{60}$ à 500 Hz (s)"), rt[:, 2]),
            (L("$T_{60}$ at 1 kHz (s)", "$T_{60}$ à 1 kHz (s)"), rt[:, 3]),
            (L("$T_{60}$ at 4 kHz (s)", "$T_{60}$ à 4 kHz (s)"), rt[:, 5])]
    body = "\n".join(f"{lab} & {num(np.percentile(v, 10))} & {num(np.median(v))} & {num(np.percentile(v, 90))} \\\\"
                     for lab, v in rows)
    tex_write("ensemble", body + "\n")
    numbers["rtMedian"] = num(np.median(rt[:, 3]))
    numbers["rtLo"] = num(np.percentile(rt[:, 3], 10))
    numbers["rtHi"] = num(np.percentile(rt[:, 3], 90))
    numbers["treated"] = f"{int(np.sum(meta['panels'][:n]))}"

    # Energy shares at 1 and 4 kHz.
    sh = an.shares(study)
    out = []
    first = L("first order", "premier ordre")
    groups = [(L("direct sound", "son direct"), [0]), (f"{first}: {surf_label('floor')}", [5]),
              (f"{first}: {surf_label('ceiling')}", [6]), (f"{first}: {surf_label('left')}", [1]),
              (f"{first}: {surf_label('right')}", [2]), (f"{first}: {surf_label('listener_wall')}", [3]),
              (f"{first}: {surf_label('speaker_wall')}", [4]),
              (L("second and third order", "deuxième et troisième ordres"), [7, 8]),
              (L("diffuse field", "champ diffus"), [9])]
    for lab, idx in groups:
        cells = []
        for fr in (1000.0, 4000.0):
            j = int(np.argmin(np.abs(f - fr)))
            v = sh[:, idx, j].sum(axis=1) * 100
            cells.append(f"{num(np.median(v), '.1f')} & {num(np.percentile(v, 10), '.1f')}--{num(np.percentile(v, 90), '.1f')}")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    tex_write("shares", "\n".join(out) + "\n")
    j = int(np.argmin(np.abs(f - 1000.0)))
    first_e = sh[:, 1:7, j].sum(axis=1)
    vert = (sh[:, 5, j] + sh[:, 6, j]) / first_e
    side = (sh[:, 1, j] + sh[:, 2, j]) / first_e
    treated = meta["panels"][:n].astype(bool)
    numbers.update(directShare=num(np.median(sh[:, 0, j]) * 100, ".1f"), firstShare=num(np.median(first_e) * 100, ".0f"),
                   lateShare=num(np.median(sh[:, 7:, j].sum(axis=1)) * 100, ".0f"),
                   vertShare=num(np.median(vert) * 100, ".0f"), vertLo=num(np.percentile(vert, 10) * 100, ".0f"),
                   vertHi=num(np.percentile(vert, 90) * 100, ".0f"), sideShare=num(np.median(side) * 100, ".0f"),
                   vertTreated=num(np.median(vert[treated]) * 100, ".0f"),
                   vertUntreated=num(np.median(vert[~treated]) * 100, ".0f"))
    early = sh[:, 0:7, j]
    early = early / early.sum(axis=1, keepdims=True)
    numbers.update(earlyDirect=num(np.median(early[:, 0]) * 100, ".0f"),
                   earlyVert=num(np.median(early[:, 5] + early[:, 6]) * 100, ".0f"))

    # Reflection directions.
    d = study.raw
    starts = d["image_starts"]
    cat = d["image_cat"][: starts[n]]
    az, el = sphere.to_angles(d["image_dirs"][: starts[n]])
    delay = d["image_delay"][: starts[n]] * 1e3
    rows = []
    cta_angles = {"floor": "$-20, -30, -40$ (V)", "ceiling": "$40, 50, 60$ (V)",
                  "left": "$\\pm 40 \\ldots \\pm 80$ (H)", "right": "$\\pm 40 \\ldots \\pm 80$ (H)",
                  "listener_wall": "$0, \\pm 10, \\pm 20, \\pm 30$ (H)", "speaker_wall": "$\\pm 90, 180$ (H)"}
    for c, surf in ((5, "floor"), (6, "ceiling"), (1, "left"), (2, "right"), (3, "listener_wall"), (4, "speaker_wall")):
        m = cat == c
        ang = el[m] if surf in ("floor", "ceiling") else az[m]
        kind = L("elevation", "élévation") if surf in ("floor", "ceiling") else L("azimuth", "azimut")
        rows.append(f"{surf_label(surf)} & {kind} & ${np.median(ang):.0f}$ & ${np.percentile(ang, 10):.0f}$ {to} "
                    f"${np.percentile(ang, 90):.0f}$ & {cta_angles[surf]} & {num(np.median(delay[m]), '.1f')} \\\\")
        numbers[f"delay{surf.replace('_', '').title()}"] = num(np.median(delay[m]), ".1f")
        numbers[f"angle{surf.replace('_', '').title()}"] = f"{np.median(ang):.0f}"
    tex_write("directions", "\n".join(rows) + "\n")

    # Weights and prediction errors.
    w = an.fit_weights(study)
    w2 = an.fit_weights(study, sp="sp")
    we = an.fit_weights(study, which="early")
    numbers.update(wLw=num(w[0]), wEr=num(w[1]), wSp=num(w[2]), wLwTwo=num(w2[0]), wErTwo=num(w2[1]),
                   wSpTwo=num(w2[2]), wLwEarly=num(we[0]), wErEarly=num(we[1]), wSpEarly=num(we[2]))
    preds = [(L("CTA-2034 PIR, $0.12/0.44/0.44$, two orbits", "PIR CTA-2034, $0,12/0,44/0,44$, deux orbites"),
              (0.12, 0.44, 0.44), "sp"),
             (L("same weights, sound power of the sphere", "mêmes poids, puissance sur la sphère"), (0.12, 0.44, 0.44),
              "sp_sphere"),
             (L("fitted, two orbits", "ajustés, deux orbites") + f", ${num(w2[0])}/{num(w2[1])}/{num(w2[2])}$", tuple(w2),
              "sp"),
             (L("fitted, sphere", "ajustés, sphère") + f", ${num(w[0])}/{num(w[1])}/{num(w[2])}$", tuple(w), "sp_sphere")]
    m = an.band(f, an.ERROR_BAND)
    rows = []
    for lab, weights, sp in preds:
        rms_all, bias_all = [], []
        for v in study.variants:
            rms, bias = an.errors(study, v, an.pir(study.curves(v), weights, sp))
            rms_all.append(np.median(rms))
            bias_all.append(np.abs(bias[m]).max())
        rows.append(f"{lab} & {num(np.min(rms_all))}--{num(np.max(rms_all))} & "
                    f"{num(np.min(bias_all))}--{num(np.max(bias_all))} \\\\")
    floor = []
    for v in study.variants:
        rms, _ = an.errors(study, v, study.rooms(v, "steady", study.train()).mean(axis=0))
        floor.append(np.median(rms))
    rows.append(L("mean of the 150 training rooms (room-to-room floor)",
                  "moyenne des 150 pièces d’apprentissage (plancher entre pièces)")
                + f" & {num(np.min(floor))}--{num(np.max(floor))} & -- \\\\")
    tex_write("errors", "\n".join(rows) + "\n")

    # Two orbits against the sphere, at the crossover.
    jx = int(np.argmin(np.abs(f - 2000.0)))
    for kind in crossover.KINDS:
        c = study.curves(name("two_way", "upright", kind))
        word = {"lr2": "LrTwo", "lr4": "LrFour", "lr8": "LrEight", "fir": "Fir"}[kind]
        numbers[f"spBias{word}"] = num(db10(c["sp"][jx]) - db10(c["sp_sphere"][jx]), ".1f")

    # Averaged curves: deviation within an octave of the crossover, and the deepest dip.
    rows = []
    vals_all = {}
    for layout in ("two_way", "two_way_wg"):
        for orient in ("upright", "side"):
            for kind in crossover.KINDS:
                v = name(layout, orient, kind)
                c = study.curves(v)
                pf = an.pir(c, w, "sp_sphere")
                steady = study.rooms(v, "steady", study.random())
                early_r = study.rooms(v, "early", study.random())
                vals = [an.local_anomaly(f, c["pir"]), an.local_anomaly(f, pf),
                        float(np.median(an.local_anomaly(f, steady))), float(np.median(an.local_anomaly(f, early_r))),
                        an.notch_depth(f, c["pir"]), float(np.median(an.notch_depth(f, steady)))]
                vals_all[v] = vals
                orient_l = L("upright", "debout") if orient == "upright" else L("on its side", "couchée")
                rows.append(f"{layout_label(layout).split(' ')[0] if layout == 'two_way' else L('waveguide', 'guide d’onde')}, "
                            f"{orient_l}, {kind_label(kind)} & " + " & ".join(num(x) for x in vals) + " \\\\")
            rows.append("\\addlinespace")
    tex_write("anomalies", "\n".join(rows[:-1]) + "\n")
    for layout, tag in (("two_way", "Dome"), ("two_way_wg", "Wg")):
        for kind, word in (("lr2", "LrTwo"), ("lr4", "LrFour"), ("lr8", "LrEight"), ("fir", "Fir")):
            vals = vals_all[name(layout, "upright", kind)]
            for i, key in enumerate(("Pir", "Fit", "Room", "Early", "PirDip", "RoomDip")):
                numbers[f"curve{tag}{word}{key}"] = num(vals[i])

    # Colouration of the paths.
    rows = []
    col_all = {}
    for layout in ("two_way", "two_way_wg"):
        for kind in crossover.KINDS:
            v = name(layout, "upright", kind)
            vals = colouration_values(study, v, w)
            dip = float(np.median(an.path_anomaly(study, v, "vertical", metric="notch")))
            col_all[v] = vals + [dip]
            short = L("dome", "dôme") if layout == "two_way" else L("waveguide", "guide d’onde")
            rows.append(f"{short}, {kind_label(kind)} & " + " & ".join(num(x) for x in vals)
                        + f" & {num(dip, '.1f')} \\\\")
        rows.append("\\addlinespace")
    tex_write("colouration", "\n".join(rows[:-1]) + "\n")
    for layout, tag in (("two_way", "Dome"), ("two_way_wg", "Wg")):
        base = col_all[name(layout, "upright", "lr4")]
        for kind, word in (("lr2", "LrTwo"), ("lr4", "LrFour"), ("lr8", "LrEight"), ("fir", "Fir")):
            vals = col_all[name(layout, "upright", kind)]
            for i, key in enumerate(("Pir", "Fit", "Steady", "Refl", "Vert", "VertDip")):
                numbers[f"col{tag}{word}{key}"] = num(vals[i], ".1f" if key == "VertDip" else ".2f")
                if kind == "fir":
                    numbers[f"col{tag}Red{key}"] = num((1 - vals[i] / base[i]) * 100, ".0f")
            if kind == "lr2":
                for i, key in enumerate(("Pir", "Fit", "Steady", "Refl", "Vert")):
                    numbers[f"col{tag}LrTwoMore{key}"] = num((vals[i] / base[i] - 1) * 100, ".0f")

    # Steady-state level of the PIR above the rooms.
    v = name("two_way", "upright", "lr4")
    _, bias = an.errors(study, v, study.curves(v)["pir"], rooms=study.random())
    numbers["pirBiasMax"] = num(bias[m].max(), ".1f")
    numbers["pirBiasLow"] = num(np.interp(300.0, f, bias), ".1f")

    tex_write("numbers", "\n".join(f"\\newcommand{{\\n{k}}}{{{val}}}" for k, val in sorted(numbers.items())) + "\n")
    return numbers


def main():
    global LANG, FIG, TAB
    if len(sys.argv) > 1 and sys.argv[1] == "fr":
        LANG = "fr"
        FIG = ROOT / "paper" / "figures" / "fr"
        TAB = ROOT / "paper" / "tables" / "fr"
    ps.apply()
    if LANG == "fr":
        locale.setlocale(locale.LC_NUMERIC, "fr_FR.UTF-8")
        plt.rcParams["axes.formatter.use_locale"] = True
    study = an.load()
    numbers = {}
    fig_setup()
    fig_kernel_maps(study)
    fig_shares(study)
    fig_reflection_angles(study)
    fig_spin(study, "two_way", "spin_dome")
    fig_spin(study, "two_way_wg", "spin_waveguide")
    fig_sp_orbits(study)
    fig_errors(study)
    fig_rotation(study)
    fig_crossovers(study)
    fig_paths(study, path_spectra(), numbers)
    fig_vertical_colouration(study)
    fig_colouration(study)
    fig_polar_maps()
    numbers = tables(study, numbers)
    for k, v in sorted(numbers.items()):
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
