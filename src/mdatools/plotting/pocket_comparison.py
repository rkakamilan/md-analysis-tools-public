"""Pocket comparison visualization.

Generates a side-by-side figure:
  left panel  – 2D structure with mean d_min halos (±SD shown in label)
  right panel – sortable metrics table (d_min, cone_dist, hydrophob, n_NO)

Requires: ``pip install 'mdatools[pocket]'`` (rdkit, cairosvg, Pillow).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

from .pocket_profile import (
    DMIN_CMAP,
    dmin_color,
    hydrophob_border,
    render_svg_and_positions,
    _load_font,
    _draw_text,
    _text_w,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

SVG_W, SVG_H = 1200, 900
SVG_SCALE    = 2.0
MOL_W        = int(SVG_W * SVG_SCALE)   # 2400
MOL_H        = int(SVG_H * SVG_SCALE)   # 1800
TBL_W        = 1400                      # right panel width
IMG_W        = MOL_W + TBL_W            # 3800
TITLE_H      = 95
HALO_R       = 22

TABLE_COL_W     = [120, 70, 150, 90, 150, 90, 95, 70]  # total=835 < TBL_W
TABLE_HEADERS   = ["Atom", "Elem", "d_min (Å)", "±SD", "cone (Å)", "±SD", "hydrophob", "n_NO"]
ROW_H           = 28
LABEL_SKIP_ELEM = {"F", "CL", "BR"}   # skip text label for terminal halogens


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------

def plot_pocket_comparison(
    mol,
    name2idx: dict[str, int],
    comparison: "pd.DataFrame",  # noqa: F821
    output_path: Path,
    *,
    title: str = "Pocket Environment Profile",
) -> Path:
    """Generate a 2D-structure + metrics-table comparison figure.

    Parameters
    ----------
    mol:
        RDKit Mol with 2D coordinates.
    name2idx:
        Atom name → RDKit atom index mapping (from
        :func:`~mdatools.pocket.profiler.build_2d_mol`).
    comparison:
        DataFrame from :meth:`~mdatools.pocket.comparator.PocketComparator.run`.
        Must contain ``d_min_mean``, ``d_min_std``, ``cone_dist_mean``,
        ``cone_dist_std``, ``hydrophob_mean``, ``n_NO_mean``.
    output_path:
        Output PNG file path.
    title:
        Title drawn at the top of the canvas.

    Returns
    -------
    Path to the written PNG.
    """
    import cairosvg
    from PIL import Image, ImageDraw

    # ── SVG → PNG ─────────────────────────────────────────────────────────
    svg_text, atom_pos_svg = render_svg_and_positions(mol, SVG_W, SVG_H)
    mol_png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=MOL_W,
        output_height=MOL_H,
    )
    mol_img = Image.open(io.BytesIO(mol_png_bytes)).convert("RGBA")

    # ── Build canvas ──────────────────────────────────────────────────────
    n_rows = sum(1 for atom in comparison.index if atom in name2idx)
    tbl_h  = (n_rows + 2) * ROW_H + 40
    body_h = max(MOL_H, tbl_h + 20)
    legend_h = 100
    total_h  = TITLE_H + body_h + legend_h + 20

    canvas = Image.new("RGB", (IMG_W, total_h), (255, 255, 255))
    canvas.paste(mol_img.convert("RGB"), (0, TITLE_H), mol_img)

    # right panel background
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([MOL_W, TITLE_H, IMG_W, TITLE_H + body_h], fill=(248, 249, 252))

    # ── Fonts ─────────────────────────────────────────────────────────────
    F_ttl  = _load_font(54)
    F_tjp  = _load_font(52, jp=True)
    F_lbl  = _load_font(24)
    F_ljp  = _load_font(22, jp=True)
    F_sm   = _load_font(20)
    F_smjp = _load_font(19, jp=True)
    F_tbl  = _load_font(20)
    F_tbjp = _load_font(18, jp=True)

    _draw_text(draw, (40, 18), title, F_ttl, F_tjp, fill=(15, 15, 55))

    # ── Halo layer ────────────────────────────────────────────────────────
    halo_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    hl = ImageDraw.Draw(halo_layer)

    for atom, row in comparison.iterrows():
        if atom not in name2idx:
            continue
        idx = name2idx[atom]
        if idx not in atom_pos_svg:
            continue
        sx, sy = atom_pos_svg[idx]
        px = sx * SVG_SCALE
        py = sy * SVG_SCALE + TITLE_H

        d    = row["d_min_mean"]
        dsd  = row.get("d_min_std", 0.0)
        h    = row.get("hydrophob_mean", 0.5)
        elem = str(row.get("element", "C"))

        fc = dmin_color(d)
        bc = hydrophob_border(h)

        hl.ellipse(
            [px - HALO_R, py - HALO_R, px + HALO_R, py + HALO_R],
            fill=(*fc, 80),
            outline=(*bc, 200),
            width=2,
        )

        if elem.upper() not in LABEL_SKIP_ELEM:
            val_str = f"{d:.1f}"
            tw = _text_w(val_str, F_lbl, F_ljp)
            tx = px - tw / 2
            ty = py + HALO_R + 1
            hl.rectangle([tx - 1, ty, tx + tw + 1, ty + 24], fill=(255, 255, 255, 190))

    # alpha-composite halos
    canvas = Image.alpha_composite(canvas.convert("RGBA"), halo_layer).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ── d_min text (after composite) ─────────────────────────────────────
    for atom, row in comparison.iterrows():
        if atom not in name2idx:
            continue
        idx = name2idx[atom]
        if idx not in atom_pos_svg:
            continue
        sx, sy = atom_pos_svg[idx]
        px = sx * SVG_SCALE
        py = sy * SVG_SCALE + TITLE_H

        d    = row["d_min_mean"]
        elem = str(row.get("element", "C"))

        if elem.upper() in LABEL_SKIP_ELEM:
            continue

        fc = dmin_color(d)
        val_str = f"{d:.1f}"
        tw = _text_w(val_str, F_lbl, F_ljp)
        _draw_text(draw, (px - tw / 2, py + HALO_R + 1), val_str, F_lbl, F_ljp, fill=tuple(fc))

    # ── Metrics table (right panel) ───────────────────────────────────────
    tbl_x0 = MOL_W + 20
    tbl_y0 = TITLE_H + 20

    _draw_text(draw, (tbl_x0, tbl_y0), "Pocket Metrics Table (sorted by d_min)", F_lbl, F_ljp, fill=(20, 20, 60))

    hdr_y = tbl_y0 + ROW_H + 6
    cx = tbl_x0
    for hh, cw in zip(TABLE_HEADERS, TABLE_COL_W):
        draw.rectangle([cx, hdr_y, cx + cw, hdr_y + ROW_H], fill=(210, 225, 245))
        _draw_text(draw, (cx + 4, hdr_y + 4), hh, F_tbl, F_tbjp, fill=(20, 20, 80))
        draw.line([cx + cw, hdr_y, cx + cw, hdr_y + ROW_H], fill=(170, 185, 210), width=1)
        cx += cw

    tbl_y = hdr_y + ROW_H
    sorted_cmp = comparison.sort_values("d_min_mean")

    for ri, (atom, row) in enumerate(sorted_cmp.iterrows()):
        if atom not in name2idx:
            continue
        bg = (255, 255, 255) if ri % 2 == 0 else (242, 246, 252)

        d   = row["d_min_mean"]
        dsd = row.get("d_min_std", 0.0)
        cd  = row.get("cone_dist_mean", 0.0)
        csd = row.get("cone_dist_std", 0.0)
        hy  = row.get("hydrophob_mean", 0.0)
        no  = row.get("n_NO_mean", 0.0)
        elem = str(row.get("element", "?"))

        fc  = dmin_color(d)
        bc  = hydrophob_border(hy)
        noc = (30, 100, 200) if no >= 2 else (40, 40, 40)

        row_vals = [
            (str(atom),       (40, 40, 40)),
            (elem,            (60, 60, 60)),
            (f"{d:.2f}",      fc),
            (f"±{dsd:.2f}",   (110, 110, 110)),
            (f"{cd:.2f}",     (60, 60, 140)),
            (f"±{csd:.2f}",   (110, 110, 110)),
            (f"{hy:.2f}",     bc),
            (f"{no:.0f}",     noc),
        ]
        cx = tbl_x0
        for (val, color), cw in zip(row_vals, TABLE_COL_W):
            draw.rectangle([cx, tbl_y, cx + cw, tbl_y + ROW_H], fill=bg)
            _draw_text(draw, (cx + 4, tbl_y + 4), val, F_tbl, F_tbjp, fill=color)
            draw.line([cx + cw, tbl_y, cx + cw, tbl_y + ROW_H], fill=(215, 220, 230), width=1)
            cx += cw
        draw.line([tbl_x0, tbl_y + ROW_H, cx, tbl_y + ROW_H], fill=(215, 220, 230), width=1)
        tbl_y += ROW_H

    # table border
    tbl_total_w = sum(TABLE_COL_W)
    draw.rectangle([tbl_x0, hdr_y, tbl_x0 + tbl_total_w, tbl_y], outline=(140, 155, 185), width=2)

    # table footnotes
    note_y = tbl_y + 10
    notes = [
        "d_min: distance to nearest protein heavy atom (Å)  ±SD across snapshots",
        "cone: min distance within ±30° outward cone (Å)",
        "hydrophob: fraction of C atoms within 5 Å sphere",
        "n_NO: count of N/O atoms within 5 Å (H-bond potential)",
    ]
    for ni, note in enumerate(notes):
        _draw_text(draw, (tbl_x0, note_y + ni * 20), note, F_sm, F_smjp, fill=(90, 90, 100))

    # ── Color-bar legend (bottom) ─────────────────────────────────────────
    legend_y = TITLE_H + body_h + 10
    _draw_text(draw, (40, legend_y + 4), "d_min scale:", F_lbl, F_ljp, fill=(30, 30, 30))
    segs = [("<=3.5A", 3.0), ("3.5-4.5", 4.0), ("4.5-5.5", 5.0), ("5.5-7.0", 6.0), (">7.0A", 8.0)]
    cx0 = 250
    for si, (sl, sv) in enumerate(segs):
        x0 = cx0 + si * 200
        fc = dmin_color(sv)
        draw.rectangle([x0, legend_y, x0 + 160, legend_y + 28], fill=fc)
        tw = _text_w(sl, F_sm, F_smjp)
        _draw_text(draw, (x0 + (160 - tw) // 2, legend_y + 30), sl, F_sm, F_smjp, fill=(40, 40, 40))

    env_x = cx0 + 5 * 200 + 50
    _draw_text(draw, (env_x, legend_y), "border=env:", F_lbl, F_ljp, fill=(30, 30, 30))
    for ei, (ec, et) in enumerate([
        ((200, 100, 20), "hydrophobic"),
        ((120, 120, 120), "mixed"),
        ((30, 100, 200), "polar"),
    ]):
        ex = env_x + 120 + ei * 220
        ey = legend_y + 4
        draw.rounded_rectangle([ex, ey, ex + 22, ey + 22], radius=3, fill=(250, 250, 250), outline=ec, width=3)
        _draw_text(draw, (ex + 28, ey + 3), et, F_sm, F_smjp, fill=(40, 40, 40))

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), dpi=(300, 300))
    logger.info("Comparison PNG → %s  (%dx%dpx)", output_path.name, canvas.width, canvas.height)
    return output_path
