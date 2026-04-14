"""Visualization for pocket environment profiles.

Generates a 2D structure diagram overlaid with per-atom d_min halos,
colour-coded by tightness (d_min) and environment polarity (hydrophob).

Requires: ``pip install 'mdatools[pocket]'`` (rdkit, cairosvg, Pillow).
"""

from __future__ import annotations

import io
import re
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

# d_min → RGB (red = tight, green = open)
DMIN_CMAP: list[tuple[float, tuple[int, int, int]]] = [
    (3.5, (210,  30,  30)),
    (4.5, (240, 120,  20)),
    (5.5, (210, 190,  20)),
    (7.0, (110, 200,  50)),
    (999, ( 40, 190,  40)),
]


def dmin_color(d: float) -> tuple[int, int, int]:
    """Map a d_min value to an RGB colour."""
    for thresh, rgb in DMIN_CMAP:
        if d <= thresh:
            return rgb
    return DMIN_CMAP[-1][1]


def hydrophob_border(h: float) -> tuple[int, int, int]:
    """Map hydrophobicity fraction to halo-border RGB colour."""
    if h >= 0.65:
        return (200, 100,  20)   # orange – hydrophobic
    elif h <= 0.40:
        return ( 30, 100, 200)   # blue – polar
    else:
        return (120, 120, 120)   # grey – mixed


# ---------------------------------------------------------------------------
# SVG rendering helpers
# ---------------------------------------------------------------------------

def render_svg_and_positions(mol, svg_w: int = 1200, svg_h: int = 800):
    """Render *mol* to SVG and extract per-atom pixel coordinates.

    Parameters
    ----------
    mol:
        RDKit Mol with 2D coordinates.
    svg_w, svg_h:
        Canvas size in pixels.

    Returns
    -------
    svg_text:
        SVG string.
    atom_pos:
        Dict mapping atom index → ``(x, y)`` pixel position.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(svg_w, svg_h)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True
    opts.padding = 0.10
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    coord_accum: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for m in re.finditer(
        r"class='bond-\d+ atom-(\d+) atom-(\d+)' d='M ([\d.]+),([\d.]+) L ([\d.]+),([\d.]+)'",
        svg,
    ):
        a1, a2 = int(m.group(1)), int(m.group(2))
        coord_accum[a1].append((float(m.group(3)), float(m.group(4))))
        coord_accum[a2].append((float(m.group(5)), float(m.group(6))))

    atom_pos = {
        idx: (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )
        for idx, pts in coord_accum.items()
    }
    return svg, atom_pos


# ---------------------------------------------------------------------------
# Font helpers (PIL)
# ---------------------------------------------------------------------------

def _load_font(size: int, jp: bool = False):
    """Load a TrueType font, falling back to PIL default."""
    from PIL import ImageFont

    candidates = (
        [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        ]
        if jp
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (0x3000 <= cp <= 0x9FFF) or (0xF900 <= cp <= 0xFAFF) or (0xFF00 <= cp <= 0xFFEF)


def _draw_text(draw, pos, text: str, fl, fj, fill=(20, 20, 20)) -> None:
    x, y = pos
    for ch in text:
        f = fj if _is_cjk(ch) else fl
        bb = f.getbbox(ch)
        draw.text((x, y), ch, fill=fill, font=f)
        x += bb[2] - bb[0]


def _text_w(text: str, fl, fj) -> float:
    w = 0.0
    for ch in text:
        f = fj if _is_cjk(ch) else fl
        bb = f.getbbox(ch)
        w += bb[2] - bb[0]
    return w


# ---------------------------------------------------------------------------
# Main PNG generator
# ---------------------------------------------------------------------------

SVG_W, SVG_H = 1200, 800
IMG_SCALE = 2.0
HALO_R = 16


def make_snapshot_png(
    mol,
    name2idx: dict[str, int],
    metrics: list[dict],
    out_path: Path,
    title: str = "",
) -> Path:
    """Generate a 2D structure + d_min halo overlay PNG.

    Parameters
    ----------
    mol:
        RDKit Mol with 2D coordinates (Compute2DCoords already called).
    name2idx:
        Mapping from atom name → RDKit atom index (from :func:`build_2d_mol`).
    metrics:
        List of per-atom dicts from :func:`compute_metrics`.
    out_path:
        Output path for the PNG file.
    title:
        Title string drawn at the top of the canvas.

    Returns
    -------
    Path to the written PNG file.
    """
    import cairosvg
    from PIL import Image, ImageDraw

    W = int(SVG_W * IMG_SCALE)
    H = int(SVG_H * IMG_SCALE)
    TH = 80   # title height (px)
    LH = 60   # legend height (px)

    svg_text, atom_pos_svg = render_svg_and_positions(mol, SVG_W, SVG_H)
    mol_png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=W,
        output_height=H,
    )
    mol_img = Image.open(io.BytesIO(mol_png_bytes)).convert("RGBA")

    canvas = Image.new("RGB", (W, TH + H + LH), (255, 255, 255))
    canvas.paste(mol_img.convert("RGB"), (0, TH), mol_img)

    F_ttl = _load_font(42)
    F_tjp = _load_font(40, jp=True)
    F_leg = _load_font(19)
    F_ljp = _load_font(18, jp=True)
    F_name = _load_font(14)
    F_dmin = _load_font(12)

    draw = ImageDraw.Draw(canvas)

    # Title
    _draw_text(draw, (30, 16), title or "Pocket environment", F_ttl, F_tjp, fill=(15, 15, 55))

    met_by_name = {m["atom"]: m for m in metrics}

    # Collect pixel centroids for outward direction computation
    px_list, py_list = [], []
    for name in met_by_name:
        if name not in name2idx or name2idx[name] not in atom_pos_svg:
            continue
        sx, sy = atom_pos_svg[name2idx[name]]
        px_list.append(sx * IMG_SCALE)
        py_list.append(sy * IMG_SCALE + TH)
    mol_cx = float(np.mean(px_list)) if px_list else W / 2
    mol_cy = float(np.mean(py_list)) if py_list else H / 2 + TH

    # Halo layer
    halo_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    hl = ImageDraw.Draw(halo_layer)

    LABEL_OFFSET = HALO_R + 36
    label_positions: dict[str, tuple] = {}

    for name, m in met_by_name.items():
        if name not in name2idx:
            continue
        idx = name2idx[name]
        if idx not in atom_pos_svg:
            continue
        sx, sy = atom_pos_svg[idx]
        px = sx * IMG_SCALE
        py = sy * IMG_SCALE + TH

        d = m["d_min"]
        h_val = m["hydrophob"]
        fc = dmin_color(d)
        bc = hydrophob_border(h_val)

        hl.ellipse(
            [px - HALO_R, py - HALO_R, px + HALO_R, py + HALO_R],
            fill=(*fc, 90),
            outline=(*bc, 220),
            width=2,
        )

        dx, dy = px - mol_cx, py - mol_cy
        dist = max(float(np.sqrt(dx * dx + dy * dy)), 1.0)
        ux, uy = dx / dist, dy / dist

        lx = px + ux * LABEL_OFFSET
        ly = py + uy * LABEL_OFFSET

        name_str = name
        dmin_str = f"{d:.1f}"
        tw_n = _text_w(name_str, F_name, F_name)
        tw_d = _text_w(dmin_str, F_dmin, F_dmin)
        box_w = max(tw_n, tw_d) + 6
        box_h = 26

        bx0 = lx - box_w / 2
        by0 = ly - box_h / 2

        hl.rounded_rectangle(
            [bx0, by0, bx0 + box_w, by0 + box_h],
            radius=3,
            fill=(255, 255, 255, 210),
            outline=(*fc, 200),
            width=1,
        )
        hl.line(
            [px + ux * HALO_R, py + uy * HALO_R,
             lx - ux * (box_w / 2 + 1), ly - uy * (box_h / 2 + 1)],
            fill=(160, 160, 160, 170),
            width=1,
        )

        label_positions[name] = (px, py, lx, ly, box_w, box_h, bx0, by0, fc)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), halo_layer).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # Atom label text
    for name, m in met_by_name.items():
        if name not in label_positions:
            continue
        _, _, lx, ly, box_w, box_h, bx0, by0, fc = label_positions[name]
        d = m["d_min"]
        name_str = name
        dmin_str = f"{d:.1f}"
        tw_d = _text_w(dmin_str, F_dmin, F_dmin)
        _draw_text(draw, (bx0 + 3, by0 + 1),  name_str, F_name, F_name, fill=(20, 20, 20))
        dx_align = box_w - tw_d - 4
        _draw_text(draw, (bx0 + dx_align, by0 + 13), dmin_str, F_dmin, F_dmin, fill=tuple(fc))

    # d_min color bar legend
    leg_y = TH + H + 8
    _draw_text(draw, (20, leg_y + 5), "d_min:", F_leg, F_ljp, fill=(40, 40, 40))
    segs = [("<=3.5A", 3.0), ("3.5-4.5", 4.0), ("4.5-5.5", 5.0), ("5.5-7.0", 6.0), (">7.0A", 8.0)]
    for si, (sl, sv) in enumerate(segs):
        x0 = 110 + si * 200
        fc = dmin_color(sv)
        draw.rectangle([x0, leg_y, x0 + 160, leg_y + 26], fill=fc)
        tw = _text_w(sl, F_leg, F_leg)
        _draw_text(draw, (x0 + (160 - tw) // 2, leg_y + 28), sl, F_leg, F_leg, fill=(50, 50, 50))

    # Border colour legend
    env_x = 110 + 5 * 200 + 30
    _draw_text(draw, (env_x, leg_y + 3), "border=env:", F_leg, F_leg, fill=(40, 40, 40))
    for ei, (ec, et) in enumerate([
        ((200, 100, 20), "hydrophobic"),
        ((120, 120, 120), "mixed"),
        ((30, 100, 200), "polar"),
    ]):
        ex = env_x + 100 + ei * 190
        draw.rounded_rectangle(
            [ex, leg_y + 2, ex + 20, leg_y + 22],
            radius=3,
            fill=(250, 250, 250),
            outline=ec,
            width=3,
        )
        _draw_text(draw, (ex + 26, leg_y + 4), et, F_leg, F_leg, fill=(40, 40, 40))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), dpi=(300, 300))
    logger.info("PNG → %s  (%dx%dpx)", out_path.name, canvas.width, canvas.height)
    return out_path
