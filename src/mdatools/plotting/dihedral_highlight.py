"""2D structure visualization with rotatable bond dihedral highlights.

Generates a PNG with the ligand's 2D structure and colour-coded bond
overlays showing the conformational flexibility (Δ° range) of each
rotatable bond.

Requires: ``pip install 'mdatools[pocket]'`` (rdkit, cairosvg, Pillow).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------

def dihedral_range_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dihedral statistics from a DihedralAnalyzer result.

    Parameters
    ----------
    df:
        DataFrame from :meth:`DihedralAnalyzer.run` (columns: ``frame``,
        ``time_ns``, then one column per dihedral label).

    Returns
    -------
    DataFrame with one row per dihedral and columns:
    ``dihedral``, ``mean``, ``std``, ``min``, ``max``, ``range``.
    """
    meta = {"frame", "time_ns", "has_hbond"}
    dihedral_cols = [c for c in df.columns if c not in meta]

    rows = []
    for col in dihedral_cols:
        series = df[col].dropna()
        rows.append({
            "dihedral": col,
            "mean":  float(series.mean()),
            "std":   float(series.std()),
            "min":   float(series.min()),
            "max":   float(series.max()),
            "range": float(series.max() - series.min()),
        })
    return pd.DataFrame(rows, columns=["dihedral", "mean", "std", "min", "max", "range"])


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

# Range (°) → RGB  (small range = green, large range = red)
_RANGE_CMAP: list[tuple[float, tuple[int, int, int]]] = [
    ( 30, ( 50, 180,  50)),   # green  — rigid
    ( 60, (200, 190,  30)),   # yellow — moderate
    (120, (230, 100,  20)),   # orange — flexible
    (999, (210,  30,  30)),   # red    — very flexible
]


def range_color(delta_deg: float) -> tuple[int, int, int]:
    """Map a dihedral range (°) to an RGB colour."""
    for thresh, rgb in _RANGE_CMAP:
        if delta_deg <= thresh:
            return rgb
    return _RANGE_CMAP[-1][1]


# ---------------------------------------------------------------------------
# Main PNG generator
# ---------------------------------------------------------------------------

SVG_W, SVG_H = 1200, 800


def plot_dihedral_highlight(
    pdb_path: Path,
    highlight_bonds: list[dict],
    output_png: Path,
    *,
    ligand_resname: str = "UNK",
    image_width: int = 1600,
) -> Path:
    """Generate a 2D structure PNG with highlighted rotatable bonds.

    Parameters
    ----------
    pdb_path:
        Path to a PDB file containing the ligand.
    highlight_bonds:
        List of dicts, each with keys:

        * ``atom1`` / ``atom2`` — atom *names* of the bond endpoints
        * ``label`` — text to draw on the bond (e.g. ``"Δ22.6°"``)
        * ``color`` — hex string (``"#E05C5C"``) **or** RGB 3-tuple

    output_png:
        Destination PNG path.
    ligand_resname:
        Residue name of the ligand in the PDB (default ``"UNK"``).
    image_width:
        Output image width in pixels.

    Returns
    -------
    Path to the written PNG.
    """
    import cairosvg
    from PIL import Image, ImageDraw

    from ..pocket.profiler import build_2d_mol
    from .pocket_profile import (
        render_svg_and_positions,
        _load_font,
        _draw_text,
        _text_w,
    )

    pdb_path = Path(pdb_path)
    output_png = Path(output_png)

    # ------------------------------------------------------------------ #
    # 1. Build 2-D mol and SVG                                             #
    # ------------------------------------------------------------------ #
    mol, name2idx = build_2d_mol(pdb_path, ligand_resname=ligand_resname)

    img_scale = image_width / SVG_W
    img_h = int(SVG_H * img_scale)
    TH = 80   # title bar height
    LH = 70   # legend height

    svg_text, atom_pos_svg = render_svg_and_positions(mol, SVG_W, SVG_H)
    mol_png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=image_width,
        output_height=img_h,
    )
    mol_img = Image.open(io.BytesIO(mol_png_bytes)).convert("RGBA")

    canvas = Image.new("RGB", (image_width, TH + img_h + LH), (255, 255, 255))
    canvas.paste(mol_img.convert("RGB"), (0, TH), mol_img)

    # ------------------------------------------------------------------ #
    # 2. Bond highlight layer                                              #
    # ------------------------------------------------------------------ #
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ol = ImageDraw.Draw(overlay)

    F_ttl  = _load_font(int(42 * img_scale / 2))
    F_ttjp = _load_font(int(40 * img_scale / 2), jp=True)
    F_lbl  = _load_font(int(22 * img_scale / 2))
    F_lbjp = _load_font(int(20 * img_scale / 2), jp=True)
    F_leg  = _load_font(int(19 * img_scale / 2))
    F_ljp  = _load_font(int(18 * img_scale / 2), jp=True)

    def _px(svg_x: float, svg_y: float) -> tuple[float, float]:
        return svg_x * img_scale, svg_y * img_scale + TH

    def _parse_color(c) -> tuple[int, int, int]:
        if isinstance(c, (list, tuple)):
            return tuple(int(v) for v in c[:3])
        s = str(c).strip().lstrip("#")
        if len(s) == 6:
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (200, 30, 30)

    LINE_W = max(4, int(8 * img_scale / 2))

    rendered_bonds = []
    for bond in highlight_bonds:
        a1_name = bond.get("atom1", "")
        a2_name = bond.get("atom2", "")
        label   = bond.get("label", "")
        color   = _parse_color(bond.get("color", "#888888"))

        idx1 = name2idx.get(a1_name)
        idx2 = name2idx.get(a2_name)

        if idx1 is None or idx2 is None:
            logger.warning("Bond (%s–%s): atom not found in 2D mol, skipping.", a1_name, a2_name)
            continue
        if idx1 not in atom_pos_svg or idx2 not in atom_pos_svg:
            logger.warning("Bond (%s–%s): no SVG position found, skipping.", a1_name, a2_name)
            continue

        px1, py1 = _px(*atom_pos_svg[idx1])
        px2, py2 = _px(*atom_pos_svg[idx2])

        # thick coloured bond
        ol.line([(px1, py1), (px2, py2)], fill=(*color, 200), width=LINE_W)
        # end caps
        R = LINE_W // 2
        for cx, cy in [(px1, py1), (px2, py2)]:
            ol.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(*color, 200))

        rendered_bonds.append((px1, py1, px2, py2, label, color))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ------------------------------------------------------------------ #
    # 3. Title                                                             #
    # ------------------------------------------------------------------ #
    _draw_text(draw, (30, 16), "Dihedral Highlight", F_ttl, F_ttjp, fill=(15, 15, 55))

    # ------------------------------------------------------------------ #
    # 4. Bond labels (drawn after composite so they sit on top)           #
    # ------------------------------------------------------------------ #
    for px1, py1, px2, py2, label, color in rendered_bonds:
        if not label:
            continue
        mx, my = (px1 + px2) / 2, (py1 + py2) / 2
        tw = _text_w(label, F_lbl, F_lbjp)
        bpad = 4
        bx0, by0 = mx - tw / 2 - bpad, my - 14
        bx1, by1 = mx + tw / 2 + bpad, my + 14
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=4,
                                fill=(255, 255, 255, 230), outline=(*color, 255), width=2)
        _draw_text(draw, (bx0 + bpad, by0 + 2), label, F_lbl, F_lbjp, fill=tuple(color))

    # ------------------------------------------------------------------ #
    # 5. Legend                                                            #
    # ------------------------------------------------------------------ #
    leg_y = TH + img_h + 8
    _draw_text(draw, (20, leg_y + 8), "Range:", F_leg, F_ljp, fill=(40, 40, 40))
    segs = [("≤30°", 15), ("30–60°", 45), ("60–120°", 90), (">120°", 150)]
    for si, (sl, sv) in enumerate(segs):
        x0 = 100 + si * 180
        fc = range_color(sv)
        BOX = 22
        draw.rounded_rectangle([x0, leg_y + 6, x0 + BOX, leg_y + 6 + BOX],
                                radius=3, fill=fc, outline=(80, 80, 80), width=1)
        _draw_text(draw, (x0 + BOX + 6, leg_y + 8), sl, F_leg, F_leg, fill=(40, 40, 40))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_png), dpi=(300, 300))
    logger.info("PNG → %s  (%dx%dpx)", output_png.name, canvas.width, canvas.height)
    return output_png
