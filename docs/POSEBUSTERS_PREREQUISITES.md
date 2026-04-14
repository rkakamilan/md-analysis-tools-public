# PoseBusters バッチ検証 — 前提条件と注意事項

## なぜ SMILES / SDF が必要なのか

PoseBusters はリガンドの **化学的妥当性**（原子価・芳香族性・立体化学）を
RDKit を通じて検証します。
MD 軌跡からエクスポートした PDB ファイルには **結合次数 (bond order) が記録されない**
ため、そのまま読み込むと以下のような問題が起きます。

| 問題 | 影響するチェック例 |
|------|-------------------|
| 芳香環がすべて単結合として解釈される | `aromatic_ring_flatness`, `double_bond_stereochemistry` |
| 共鳴構造が正しく再現されない | `formal_charges`, `valence` |
| 二重結合の立体（E/Z）が失われる | `double_bond_stereochemistry` |

> **結論**: SMILES または SDF ファイルを必ず用意し、
> `PoseBustersValidator` に渡して正しい結合次数を付与してください。

---

## 必要なもの

### 1. 追加パッケージのインストール

```bash
# pip
pip install "mdatools[posebusters] @ git+https://github.com/rkakamilan/md-analysis-tools-public.git"

# uv
uv add "mdatools[posebusters] @ git+https://github.com/rkakamilan/md-analysis-tools-public.git"
```

インストールされるパッケージ:
- `posebusters >= 0.6`
- `rdkit >= 2023.9`

### 2. リガンドの SMILES または SDF ファイル（**必須**）

| 形式 | 用途 | 入手方法の例 |
|------|------|------------|
| **SMILES 文字列** | 結合次数テンプレート | PubChem、ChEMBL、Schrödinger Maestro、MarvinSketch でコピー |
| **SDF / MOL ファイル** | 同上 | docking 入力ファイル、化合物ライブラリ、Maestro 等からエクスポート |

> **注意**: PDBQT 形式は原子電荷情報を持ちますが結合次数は不完全です。
> PDBQT を SMILES に変換する場合は Open Babel を使用してください:
>
> ```bash
> obabel input.pdbqt -O ligand.sdf
> ```

### 3. タンパク質 PDB ファイル（clean）

PoseBusters のポーズ検証には、**溶媒・イオンを除去した**タンパク質 PDB が必要です。
`FrameExporter` は自動で溶媒・イオンを除去しますが、
タンパク質単体の PDB は別途用意してください。

```
# 例: MDAnalysis で準備
from mdatools.io.writers import write_clean_pdb  # 実装予定
```

または:

```bash
# PyMOL での例
pymol -c -d "load traj.pse; remove solvent; remove (not polymer and not resname UNK); save protein.pdb, polymer"
```

---

## 使い方

```python
from pathlib import Path
from mdatools.config import AnalysisConfig
from mdatools.posebusters import FrameExporter, PoseBustersValidator
from mdatools.plotting.posebusters_plots import (
    plot_pb_pass_rate,
    plot_pb_check_heatmap,
    plot_pb_score_timeseries,
)

# ============================================================
# CONFIG — edit this section
# ============================================================
cfg = AnalysisConfig(
    ligand_resname = "UNK",
    dt_ns          = 2.0,
)

# リガンドの正しい結合次数を持つ SMILES（必須）
LIGAND_SMILES = "CC1=CC=CC=C1"   # ← あなたのリガンドの SMILES に変更

# または SDF ファイル（SMILES の代わりに使用可）
# LIGAND_SDF = Path("ligand.sdf")

# タンパク質 PDB（溶媒・イオン除去済み）
PROTEIN_PDB = Path("protein_clean.pdb")

TOPOLOGY   = Path("../run01/equilibrating_topology.pdb")
TRAJECTORY = Path("../run01/trajectory.xtc")
SAMPLE_NAME = "run01"
# ============================================================

from mdatools.universe import load_and_align

u = load_and_align(TOPOLOGY, TRAJECTORY, cfg)

# Step 1: フレームをエクスポート
exporter = FrameExporter(cfg)
pdb_files = exporter.export_from_universe(
    u,
    output_dir=Path("./pb_frames") / SAMPLE_NAME,
    prefix=f"snapshot_{SAMPLE_NAME}",
    step=1,
)

# Step 2: PoseBusters バッチ実行
validator = PoseBustersValidator(
    cfg,
    ligand_smiles=LIGAND_SMILES,   # or ligand_sdf=LIGAND_SDF
)
result = validator.run(
    pdb_dir=Path("./pb_frames") / SAMPLE_NAME,
    protein_path=PROTEIN_PDB,
)

print(f"Pass rate: {result.pass_rate:.1%}")
print(result.top_frames(10))
```

---

## よくあるエラーと対処法

### `ValueError: Either ligand_smiles or ligand_sdf must be provided`

SMILES または SDF を渡していません。
```python
# ❌ NG
validator = PoseBustersValidator(cfg)

# ✅ OK
validator = PoseBustersValidator(cfg, ligand_smiles="CC1=CC=CC=C1")
```

### `ValueError: Bond order assignment failed`

SMILES/SDF の原子接続性がトポロジーのリガンドと異なります。
確認ポイント:
- SMILES の重原子数が PDB の UNK 残基の重原子数と一致するか
- 環構造・置換基の位置が正しいか
- MDAnalysis の `u.select_atoms("resname UNK")` で原子数を確認

```python
lig = u.select_atoms("resname UNK")
print(f"重原子数: {len(lig.select_atoms('not name H*'))}")
```

### `ValueError: No HETATM records for resname 'UNK'`

リガンド残基名が `cfg.ligand_resname` と異なります。
```python
# トポロジーのリガンド残基名を確認
lig = u.select_atoms("not protein")
print(set(lig.resnames))
```

### PoseBusters チェックがほぼ全滅する

タンパク質 PDB に溶媒・イオンが含まれているか、
またはリガンドのキラル中心が間違っています。

---

## 既知の偽陽性

`non_aromatic_ring_non_flatness` は PDB 形式のフォーマット制約により
**常に False（失敗）** となります。`PoseBustersValidator` はこのチェックを
自動的に `all_pass` の計算から除外しています。

追加で除外したいチェックがある場合:
```python
validator = PoseBustersValidator(
    cfg,
    ligand_smiles=LIGAND_SMILES,
    exclude_checks=["non_aromatic_ring_non_flatness", "other_check_name"],
)
```
