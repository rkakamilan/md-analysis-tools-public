# mdatools

Generic MD analysis tools for protein-ligand simulations.

## Installation

### pip / uv

```bash
pip install mdatools
uv add mdatools
```

### バージョン固定（推奨）

```bash
pip install "mdatools==0.2.1"
uv add "mdatools==0.2.1"
```

### Optional extras

| Extra | 追加機能 | インストール例 |
|-------|---------|--------------|
| `posebusters` | PoseBusters バッチ検証（NB06） | `pip install "mdatools[posebusters] @ git+..."` |
| `pocket` | ポケット環境プロファイル・比較（NB07–08） | `pip install "mdatools[pocket] @ git+..."` |

> ⚠️ `posebusters` extra を使う際は **リガンドの SMILES または SDF** が必須です。
> 詳細は [`docs/POSEBUSTERS_PREREQUISITES.md`](docs/POSEBUSTERS_PREREQUISITES.md) を参照。

### 開発用（リポジトリを clone して editable install）

```bash
git clone https://github.com/rkakamilan/md-analysis-tools-public.git
cd md-analysis-tools-public
uv sync --extra dev          # 開発依存も含めてインストール
uv sync --extra dev --extra notebook  # Notebook 用依存も追加
```

---

## Quick start

各ノートブックの先頭 CONFIG セルだけ編集して実行します:

```python
from pathlib import Path
from mdatools.config import AnalysisConfig, ResidueGroup

cfg = AnalysisConfig(
    ligand_resname = "LIG",
    dt_ns          = 2.0,
    # residue_groups: name any set of residues you want to prioritise in
    # template selection (NB03).  When the ligand contacts one of these
    # residues, `bonus` points are added to the total score on top of the
    # H-bond quality score (max ~100 pts) and stability score (max 15 pts).
    # A bonus of 20 corresponds to roughly one "good H-bond distance" worth
    # of score — enough to promote a snapshot that hits the target region
    # without completely overriding H-bond quality.
    # Examples:
    #   GPCR   — extracellular domain (ECD) residues contacted by peptide ligands
    #   Kinase — hinge residues that anchor ATP-competitive inhibitors
    #   Any    — allosteric pocket, key salt-bridge partner, etc.
    residue_groups = {
        # GPCR ECD example
        "ECD":    ResidueGroup(resids=[35, 36, 39], bonus=20),
        # Kinase hinge example
        # "Hinge": ResidueGroup(resids=[83, 85], bonus=20),
    },
)
REPLICA_ROOTS = [Path("../run01"), Path("../run02")]
```

### Notebooks

| Notebook | 内容 | 依存 extra |
|----------|------|-----------|
| `01_rmsd.ipynb` | Backbone + ligand RMSD グリッド | — |
| `02_hbonds.ipynb` | H-bond 解析 + プロット | — |
| `03_template_selection.ipynb` | スコアリング + スナップショット抽出 | — |
| `04_validation.ipynb` | PyMOL コンフォメーションタイムライン | — |
| `05_dihedral.ipynb` | 回転可能結合の二面角解析 | — |
| `06_posebusters.ipynb` | PoseBusters バッチ検証 | `posebusters` |
| `07_pocket_profile.ipynb` | ポケット環境プロファイル（単一スナップショット） | `pocket` |
| `08_pocket_comparison.ipynb` | ポケット環境プロファイル複数比較 | `pocket` |

---

## CLI

```bash
# H-bond バッチ解析
mdatools-batch-hbonds run01/ run02/ --ligand UNK --output-dir ./hbond_results

# RMSD バッチ解析
mdatools-batch-rmsd run01/ run02/ --output-dir ./results

# PoseBusters バッチ検証
mdatools-pb-validate \
  --topology run01/equilibrating_topology.pdb \
  --trajectory run01/trajectory.xtc \
  --protein protein_clean.pdb \
  --smiles "CC1=CC=CC=C1" \
  --output-dir ./pb_results

# Movie 生成
mdatools-movie --pse run01/traj.pse --output movie.mp4 --fps 15
```

---

## API overview

### Pocket environment (NB07–08)

```python
from mdatools.pocket import PocketProfiler, PocketComparator
from mdatools.pocket.profiler import build_2d_mol
from mdatools.plotting.pocket_profile import make_snapshot_png
from mdatools.plotting.pocket_comparison import plot_pocket_comparison

# 単一スナップショット
profiler = PocketProfiler(ligand_resname="UNK")
metrics  = profiler.profile(Path("snapshot_clean.pdb"))

# 複数スナップショット比較
metrics_dict = profiler.profile_batch(pdb_paths, output_dir=Path("./results"))
comparator   = PocketComparator()
comparison   = comparator.run(metrics_dict)
print(comparator.auto_report(comparison))

# 可視化
mol, name2idx = build_2d_mol(Path("snapshot_clean.pdb"))
make_snapshot_png(mol, name2idx, metrics, out_path=Path("pocket.png"))
plot_pocket_comparison(mol, name2idx, comparison, output_path=Path("comparison.png"))
```

### PoseBusters バッチ検証 (NB06)

```python
from mdatools.posebusters import FrameExporter, PoseBustersValidator
from mdatools.config import AnalysisConfig
from mdatools.universe import load_and_align

cfg = AnalysisConfig(ligand_resname="UNK", output_dir=Path("./results"))
u   = load_and_align(Path("topology.pdb"), Path("trajectory.xtc"), cfg)

exporter  = FrameExporter(cfg)
pdb_files = exporter.export_from_universe(u, output_dir=Path("./pb_frames"), prefix="snap")

validator = PoseBustersValidator(cfg, ligand_smiles="CC1=CC=CC=C1")
result    = validator.run(pdb_dir=Path("./pb_frames"), protein_path=Path("protein_clean.pdb"))
print(f"Pass rate: {result.pass_rate:.1%}")
print(result.top_frames(10))
```

---

## Tests

```bash
uv run pytest tests/
```

---

## Project structure

```
src/mdatools/
├── config.py              # Pydantic v2 設定モデル
├── universe.py            # Universe ロード + アライメント
├── analysis/              # RMSD, H-bonds, contacts, dihedral
├── scoring/               # Template scorer + snapshot selector
├── posebusters/           # PoseBusters バッチ検証 (optional: posebusters extra)
│   ├── frame_exporter.py  #   軌跡フレーム → PDB 一括出力
│   └── batch_validator.py #   PoseBusters 全フレーム実行・集計
├── pocket/                # ポケット環境解析 (optional: pocket extra)
│   ├── profiler.py        #   d_min / hydrophob / n_NO / cone_dist 計算
│   └── comparator.py      #   複数スナップショット間の集計・考察
├── pymol_bridge/          # PML 生成, PyMOL subprocess, movie
├── io/                    # ファイル探索・CSV 入出力
├── plotting/              # Matplotlib / PIL 描画関数
│   ├── pocket_profile.py  #   2D 構造式 + d_min ハロー PNG
│   └── pocket_comparison.py # 2D 構造式 + メトリクステーブル PNG
└── cli/                   # CLI エントリポイント
```

---

## Related Tools

| Tool | 役割 |
|---|---|
| [docking-analysis-tools](https://github.com/rkakamilan/docking-analysis-tools) | ドッキング結果解析・ポーズフィルタリング・仮想スクリーニング評価 (EF / ROC-AUC) |

### 連携ワークフロー

```
mdatools                          docking-analysis-tools
─────────────────────────────     ──────────────────────────────────
MD トラジェクトリ解析
  └─ PoseClusterer
      代表コンフォメーション PDB ──► 受容体準備 → ドッキング実行
                                    ポーズフィルタリング・クラスタリング
                                    EF / ROC-AUC で最良コンフォメーション選定
上位ヒットの MD バリデーション ◄── 上位ポーズ PDB
  └─ RMSD · H-bond · コンセンサス
      最終ヒットリスト
```

アンサンブルドッキング（MD コンフォメーション → ドッキング → VS 評価）の
詳細な手順は [`docs/ensemble_docking.md`](docs/ensemble_docking.md) を参照。
