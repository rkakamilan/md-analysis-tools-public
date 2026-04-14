# Quick Start Guide

## 最小手順

各ノートブックの **先頭 CONFIG セルだけ** 編集して実行します。他のセルは変更不要です。

```python
# どのノートブックも共通のパターン
from pathlib import Path
from mdatools.config import AnalysisConfig

cfg = AnalysisConfig(
    ligand_resname = "LIG",   # ← 自分の系のリガンド残基名に変更
    dt_ns          = 2.0,     # ← フレーム間隔 (ns) に変更
)
REPLICA_ROOTS = [Path("../run01"), Path("../run02")]
```

設定パラメータの詳細は [`configuration_guide.md`](configuration_guide.md) を参照してください。

---

## ノートブック実行フロー

```
必須 / 推奨の実行順序

NB01 RMSD ──────────────────────────────────────────┐
NB02 H-bonds ──────────────────────────────────────►│
                                                      ▼
                                              NB03 Template selection
                                               （代表コンフォメーション選択）
                                                      │
                              ┌───────────────────────┼────────────────────┐
                              ▼                       ▼                    ▼
                     NB04 Validation          NB05 Dihedral        NB07/08 Pocket
                      （PyMOL で確認）          （柔軟性解析）        （環境プロファイル）
```

独立して実行できるノートブック（NB01–03 の出力不要）:

| Notebook | 入力 |
|---|---|
| NB06 PoseBusters | 軌跡ファイル + SMILES |
| NB09 Dihedral Highlight | スナップショット PDB（NB03 出力推奨） |
| NB10 RMSF | 軌跡ファイル |
| NB11 Interaction Fingerprint | 軌跡ファイル |
| NB12 Pose Clustering | 軌跡ファイル |
| NB13 Water Bridges | 軌跡ファイル |
| NB14 Convergence | 軌跡ファイル |
| NB15 Dimensionality | 軌跡ファイル |
| NB16 ADMET | SMILES のみ（軌跡不要） |
| NB17 Consensus | 他 NB の数値出力（CSV など） |

---

## ユースケース別の推奨 NB

### A. 結合安定性を素早く確認したい

```
NB01 → NB02 → NB03
```

NB01 で RMSD が安定しているか確認 → NB02 で H-bond 占有率を見る →
NB03 で代表スナップショットを PDB 出力。

### B. ドッキングの受容体として使う代表構造を取得したい（アンサンブルドッキング）

```
NB01 → NB02 → NB03 → NB12（クラスタリング）
```

NB12 `PoseClusterer` で MD コンフォメーションをクラスタリングし、
各クラスター代表 PDB をドッキングの受容体として使用します。
詳細は [`ensemble_docking.md`](ensemble_docking.md) を参照。

### C. リガンドの ADMET プロファイルを確認したい

```
NB16（軌跡不要、SMILES だけで実行可能）
```

複数リガンドの QED・SA-score・Lipinski Ro5・PAINS アラートを一括計算します。

### D. 複数の MD 指標を統合してリガンドをランキングしたい

```
NB01, NB02, NB10, NB16 → （各 CSV を収集） → NB17
```

NB17 `ConsensusRanker` に指標 DataFrame を渡すと、Borda / Z-score で
統合スコアと Pareto 最適解を出力します。

### E. 結合ポケット環境を可視化したい

```
NB03（スナップショット PDB 出力）→ NB07 → NB08（複数スナップショット比較）
```

NB07 は単一スナップショットの 2D ポケット環境図、
NB08 は複数スナップショット間の変動を比較します。
いずれも `pip install 'mdatools[pocket]'` が必要です。

### F. PoseBusters で結合ポーズの化学的妥当性を検証したい

```
NB06（軌跡 + SMILES）
```

`pip install 'mdatools[posebusters]'` と SMILES（または SDF）が必要です。
詳細は [`POSEBUSTERS_PREREQUISITES.md`](POSEBUSTERS_PREREQUISITES.md) を参照。

---

## ノートブック一覧と必要な依存

| NB | タイトル | 必須 extra |
|---|---|---|
| 01 | Backbone + Ligand RMSD | — |
| 02 | H-bond 解析 | — |
| 03 | テンプレート選択・スナップショット抽出 | — |
| 04 | PyMOL コンフォメーションタイムライン | PyMOL |
| 05 | 回転可能結合の二面角解析 | — |
| 06 | PoseBusters バッチ検証 | `posebusters` |
| 07 | ポケット環境プロファイル（単一スナップショット） | `pocket` |
| 08 | ポケット環境プロファイル比較（複数スナップショット） | `pocket` |
| 09 | 二面角柔軟性ハイライト（2D 構造図） | `pocket` |
| 10 | RMSF（残基揺らぎ） | — |
| 11 | Interaction Fingerprint（ProLIF） | `interactions` |
| 12 | Pose Clustering（Ward / DIVINE） | — |
| 13 | Water Bridges | — |
| 14 | 収束性解析（Block averaging） | — |
| 15 | PCA / TICA 次元削減 | `dimensionality` |
| 16 | ADMET / Drug-likeness | `admet` |
| 17 | Multi-metric Consensus Ranking | — |

extra のインストール例:

```bash
pip install "mdatools[pocket]"
pip install "mdatools[posebusters]"
pip install "mdatools[interactions]"
pip install "mdatools[dimensionality]"
pip install "mdatools[admet]"
# 複数まとめて
pip install "mdatools[pocket,posebusters,interactions]"
```

---

## トラブルシューティング

### `No ligand atoms found` エラー

`ligand_resname` が実際の PDB の残基名と一致していません。
```bash
grep "^HETATM" topology.pdb | awk '{print $4}' | sort -u
```
で確認してください。

### `No topology file found` エラー

`topology_glob` パターンが `REPLICA_ROOTS` 以下のファイル名と一致していません。
```bash
ls run01/
```
でファイル名を確認し、`topology_glob` を調整してください。

### RMSD が最初から大きくジャンプする

`RMSDConfig(ref_frame=0)` のデフォルトは最初のフレームを参照にします。
平衡化後のフレームを参照にしたい場合は `ref_frame=100`（フレーム番号で指定）のように変更してください。
