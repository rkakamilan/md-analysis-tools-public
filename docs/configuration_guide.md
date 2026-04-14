# Configuration Guide

`AnalysisConfig` は全ノートブックの CONFIG セル先頭で作るオブジェクトです。
Pydantic v2 モデルのため、型違いや未知フィールドは import 時にエラーとして検出されます。

> **表記について**: 本ドキュメントでは `NB01`, `NB03` などの略称を使います。
> これは `notebooks/01_rmsd.ipynb`, `notebooks/03_template_selection.ipynb` など、
> `notebooks/` ディレクトリ内のノートブックをファイル先頭の番号で指しています。

---

## AnalysisConfig — トップレベル設定

```python
from mdatools.config import AnalysisConfig

cfg = AnalysisConfig(
    ligand_resname  = "LIG",   # ← 必ず自分の系に合わせる
    dt_ns           = 2.0,
)
```

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `ligand_resname` | `str` | `"UNK"` | トポロジー PDB 内のリガンド残基名。`grep HETATM topology.pdb` で確認 |
| `topology_glob` | `str` | `"equilibrating_topology.pdb"` | `REPLICA_ROOTS` 以下でトポロジーを探す glob パターン |
| `trajectory_glob` | `str` | `"trajectory.xtc"` | 軌跡ファイルを探す glob パターン |
| `dt_ns` | `float` | `2.0` | フレーム間隔（ns）。時間軸ラベルの計算に使用。`dt_ns = stride × dt_output` |
| `output_dir` | `Path` | `Path("./results")` | CSV・PKL など数値出力の保存先 |
| `figures_dir` | `Path` | `Path("./figures")` | PNG・PDF など図の保存先 |
| `contact_cutoff` | `float` | `4.5` | 接触フィンガープリントの距離閾値（Å）。疎水性接触を含める場合は 5.0 程度 |
| `residue_groups` | `dict[str, ResidueGroup]` | `{}` | 重点残基グループ → [ResidueGroup](#residuegroup) 参照 |
| `hbond` | `HBondConfig` | デフォルト値 | 水素結合検出パラメータ → [HBondConfig](#hbondconfig) 参照 |
| `rmsd` | `RMSDConfig` | デフォルト値 | RMSD 計算パラメータ → [RMSDConfig](#rmsdconfig) 参照 |
| `scoring` | `ScoringWeights` | デフォルト値 | テンプレート選択スコアリング → [ScoringWeights](#scoringweights) 参照 |
| `n_frames` | `int \| None` | `None` | 読み込むフレーム数の上限。`None` = 全フレーム |
| `pymol_executable` | `str` | `"pymol"` | PyMOL のパス（NB04 で使用）。例: `"/opt/homebrew/bin/pymol"` |
| `ffmpeg_executable` | `str` | `"ffmpeg"` | ffmpeg のパス（movie 生成で使用） |

---

## ResidueGroup

NB03 テンプレート選択の「ボーナス加点」対象残基グループを定義します。

```python
from mdatools.config import ResidueGroup

residue_groups = {
    "Hinge": ResidueGroup(
        resids   = [83, 85],   # 残基番号リスト
        resnames = [],          # 残基名でも絞り込む場合（通常は空でよい）
        bonus    = 20.0,        # ヒット時の加点
    ),
}
```

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `resids` | `list[int]` | `[]` | 対象残基の番号リスト（トポロジーの resid と一致させること） |
| `resnames` | `list[str]` | `[]` | 追加フィルタとして残基名を指定（空 = resids のみで判定） |
| `bonus` | `float` | `20.0` | ヒット時に合計スコアへ加算される点数 |

### bonus の量感

テンプレートスコアは以下の和です:

```
total = hbond_score (最大 ~100 pt)
      + Σ group_bonus (ヒットしたグループ数 × bonus)
      + stability_score (最大 15 pt)
```

`bonus=20` は「良好な H-bond 距離」1件分に相当します。
優先度を強くしたい場合は 30–50、参考程度なら 10 以下が目安です。

---

## HBondConfig

```python
from mdatools.config import HBondConfig

cfg = AnalysisConfig(
    hbond = HBondConfig(
        d_a_cutoff   = 3.5,   # D···A 距離の上限 (Å)
        angle_cutoff = 150.0, # D-H···A 角度の下限 (°)
        start = None,         # フレーム開始（None = 最初から）
        stop  = None,         # フレーム終了（None = 最後まで）
        step  = 1,            # フレームのサンプリング間隔
    ),
)
```

| フィールド | デフォルト | 説明 |
|---|---|---|
| `d_a_cutoff` | `3.5` | ドナー–アクセプター距離の上限（Å）。厳しくしたい場合は 3.2 |
| `angle_cutoff` | `150.0` | D-H···A 角度の下限（°）。MDAnalysis のデフォルト値と同じ |
| `start` / `stop` / `step` | `None / None / 1` | MDAnalysis `run()` に渡すフレーム範囲。平衡化後のみ解析する場合に指定 |

---

## RMSDConfig

```python
from mdatools.config import RMSDConfig

cfg = AnalysisConfig(
    rmsd = RMSDConfig(
        align_select    = "protein and name CA",  # アライメント選択
        backbone_select = "backbone",             # RMSD 計算対象
        ref_frame       = 0,                      # 参照フレーム
        start = None,
        stop  = None,
        step  = 1,
    ),
)
```

| フィールド | デフォルト | 説明 |
|---|---|---|
| `align_select` | `"protein and name CA"` | トラジェクトリを重ね合わせる原子選択 |
| `backbone_select` | `"backbone"` | バックボーン RMSD の計算対象 |
| `ref_frame` | `0` | 参照フレームのインデックス（0 = 最初のフレーム） |
| `start` / `stop` / `step` | `None / None / 1` | フレーム範囲 |

---

## ScoringWeights

NB03 テンプレート選択スコアの重みを調整します。
通常はデフォルト値のままで問題ありません。

```python
from mdatools.config import ScoringWeights

cfg = AnalysisConfig(
    scoring = ScoringWeights(
        occ_weight       = 50.0,  # occupancy の最大得点
        dist_weight      = 30.0,  # 距離の最大得点
        angle_weight     = 20.0,  # 角度の最大得点
        stability_weight = 15.0,  # 後半フレーム集中度の最大得点
        dist_ref         = 3.5,   # 距離スコアの基準値 (Å)
        dist_range       = 0.7,   # dist_ref からこの幅で満点 → 0 に線形減衰
        angle_ref        = 120.0, # 角度スコアの基準値 (°)
        angle_range      = 60.0,  # angle_ref からこの幅で満点 → 0 に線形減衰
        min_frame_percentile = 0.3,  # 安定性評価の対象開始位置（0.3 = 後ろ70%）
    ),
)
```

スコア式:
```
hbond_score = clip(occ/100 * occ_weight)
            + clip((dist_ref - dist) / dist_range * dist_weight)
            + clip((angle - angle_ref) / angle_range * angle_weight)
```

---

## よく使う設定パターン

### 軌跡のサンプリングを間引く

```python
cfg = AnalysisConfig(
    hbond = HBondConfig(step=10),   # 10 フレームおきに解析
    rmsd  = RMSDConfig(step=10),
)
```

### 平衡化後のフレームだけ解析する

```python
# 全 500 フレームのうち後半 250 フレームだけ使う
cfg = AnalysisConfig(
    hbond = HBondConfig(start=250),
    rmsd  = RMSDConfig(start=250),
)
```

### リガンド残基名が PDB 標準でない場合

```python
cfg = AnalysisConfig(ligand_resname="MOL")   # Amber などでよくある
```
