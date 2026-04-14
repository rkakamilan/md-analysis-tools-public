# Ensemble Docking ガイド

**アンサンブルドッキング（Relaxed Complex Scheme）** では、
MD トラジェクトリから代表コンフォメーションを抽出してドッキングのレセプターとして使用します。
タンパク質の構造多様性を明示的に考慮できるため、高度なフレキシビリティを持つ標的（GPCRなど）に特に有効です。

本ドキュメントでは、mdatools がアンサンブルドッキングワークフローの中で担うフェーズと
[docking-analysis-tools](https://github.com/rkakamilan/docking-analysis-tools) との連携方法を説明します。

---

## mdatools が担うフェーズ

```
MD トラジェクトリ
    │
    ├─ [任意] 収束確認 ──────────────► ConvergenceAnalyzer (NB14)
    │                                  RMSD / H-bond 安定域を特定
    │
    ├─ PoseClusterer ────────────────► リガンド RMSD 行列でクラスタリング
    │   (analysis/clustering.py)       Ward 法 or k-means
    │                                  クラスター数 = ドッキング試行数
    │
    └─ extract_representatives() ────► 代表フレーム PDB
        (analysis/clustering.py)       1 クラスター 1 ファイル
                                       → docking-analysis-tools へ渡す
```

---

## ステップ 1: トラジェクトリの収束確認（推奨）

コンフォメーション選定の前に、充分にサンプリングされた領域のみを対象にします。

```python
from pathlib import Path
from mdatools.config import AnalysisConfig
from mdatools.universe import load_and_align
from mdatools.analysis.convergence import ConvergenceAnalyzer
from mdatools.plotting.convergence_plots import plot_convergence

cfg = AnalysisConfig(
    ligand_resname  = "LIG",
    topology_glob   = "topology.pdb",
    trajectory_glob = "trajectory.xtc",
    dt_ns           = 2.0,
)
u = load_and_align(Path("topology.pdb"), Path("trajectory.xtc"), cfg)

conv   = ConvergenceAnalyzer(cfg).run(u, sample_name="egfr")
fig, _ = plot_convergence(conv)
fig.savefig("convergence.png")

# 収束後の領域（例: 後半 50%）のみを対象にする場合
start_frame = len(u.trajectory) // 2
```

---

## ステップ 2: PoseClusterer でコンフォメーション選定

### 基本的な使い方

```python
from mdatools.analysis.clustering import PoseClusterer

# Ward 階層クラスタリング（デフォルト）
# n_clusters を指定しない場合は rmsd_cutoff=2.0 Å で自動分割
clusterer = PoseClusterer(cfg, method="ward", n_clusters=5)
result    = clusterer.run(u, sample_name="egfr_md")

print(result.summary)
#    cluster_id  n_frames    pct  center_frame  mean_rmsd
# 0           0        42  42.00           153       0.82
# 1           1        28  28.00           312       0.74
# 2           2        18  18.00           521       0.91
# ...
```

### パラメータの選び方

| パラメータ | 説明 | 推奨値 |
|---|---|---|
| `method` | クラスタリング手法 | `"ward"`（デフォルト・安定）/ `"kmeans"`（大規模トラジェクトリ） |
| `n_clusters` | クラスター数 = 生成する代表構造数 | 3–10（計算コストとの兼ね合い） |
| `rmsd_cutoff` | `n_clusters=None` 時の自動分割閾値 (Å) | 1.5–3.0（リガンドサイズによる） |

### クラスター数の目安

| 標的の特徴 | 推奨クラスター数 |
|---|---|
| 剛直なキナーゼ（DFG-in 優位） | 3–5 |
| DFG-in/out 両状態を含む | 5–8 |
| GPCR（大きなコンフォメーション変化） | 8–15 |
| ループが大きく動く標的 | 10–20 |

---

## ステップ 3: 代表コンフォメーションを PDB として書き出し

```python
from pathlib import Path

output_dir = Path("./conformations")
pdb_paths  = clusterer.extract_representatives(u, result, output_dir=output_dir)

for path in pdb_paths:
    print(path)
# conformations/cluster00_frame153.pdb
# conformations/cluster01_frame312.pdb
# conformations/cluster02_frame521.pdb
# ...
```

出力される PDB は:
- タンパク質 + リガンドのみ（溶媒・イオン除去済み）
- `ATOM` / `HETATM` のみの標準形式
- docking-analysis-tools の `prepare_receptor()` に直接渡せる

---

## ステップ 4: docking-analysis-tools との連携

### 受け渡し

```python
# docking-analysis-tools 側（別環境 / 別スクリプト）
from docking_analysis.preparation.receptor import prepare_receptor
from docking_analysis.preparation.gridbox import gridbox_from_residues
from docking_analysis.docking.runner import VinaRunner
from pathlib import Path

pdb_paths = sorted(Path("./conformations").glob("cluster*.pdb"))

results_by_conf = {}
for pdb in pdb_paths:
    receptor = prepare_receptor(pdb)
    box      = gridbox_from_residues(receptor, binding_site_resids=[84, 86, 110])
    runner   = VinaRunner.from_config("project_config.toml")
    results_by_conf[pdb.stem] = runner.run_batch(ligand_list, receptor, box)
```

### VS 評価でコンフォメーションを選定（Issue #87 実装後）

```python
# docking-analysis-tools 側
from docking_analysis.analysis.vs_metrics import VirtualScreeningEvaluator
from docking_analysis.plotting.vs_plots import plot_roc_curves, plot_ef_bars

evaluator = VirtualScreeningEvaluator(ef_percents=[1.0, 5.0, 10.0])
vs_results = evaluator.run_multi(
    {name: df["docking_score"].values for name, df in results_by_conf.items()},
    activity_labels,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_roc_curves(vs_results, ax=axes[0])
plot_ef_bars(vs_results, pct=5.0, ax=axes[1])
```

詳細は docking-analysis-tools の
[`docs/pipeline_integration.md`](https://github.com/rkakamilan/docking-analysis-tools/blob/main/docs/pipeline_integration.md)
を参照。

---

## ステップ 5: 選定コンフォメーションで MD バリデーション（任意）

最良コンフォメーションへのドッキングで得られた上位ヒットに対して、
MD シミュレーションを実施し mdatools で安定性を確認します。

```python
# NB01 (RMSD) + NB02 (H-bonds) + NB17 (コンセンサスランキング) のパターン
from mdatools.analysis.rmsd import RMSDAnalyzer
from mdatools.analysis.hbonds import HBondAnalyzer
from mdatools.scoring.consensus import ConsensusRanker

# 各ヒット化合物の MD 結果を集約
rmsd_results  = {name: RMSDAnalyzer(cfg).run(u, name) for name, u in universes.items()}
hbond_results = {name: HBondAnalyzer(cfg).run(u, name) for name, u in universes.items()}

# コンセンサスランキングで最終順位を決定
metrics_df = build_metrics_df(rmsd_results, hbond_results)
ranker     = ConsensusRanker(higher_is_better={"hbond_occ": True, "rmsd_mean": False})
final_rank = ranker.rank_borda(metrics_df)
```

---

## よくある質問

**Q. クラスタリングにはリガンド RMSD と backbone RMSD のどちらを使うべき？**  
A. mdatools の `PoseClusterer` はリガンド RMSD（リガンドが動く範囲でポケット形状を捉える）を使います。
タンパク質全体の構造多様性を見たい場合は RMSD 解析 (NB01) で確認してから、
スクリーニングに使うポケット周辺の柔軟性に注目してクラスター数を調整してください。

**Q. T4 lysozyme 挿入や欠損ループがある構造は使えるか？**  
A. 使えます。`extract_representatives()` は MDAnalysis の Universe をそのまま PDB に書き出すため、
挿入配列や非標準残基もそのまま含まれます。ただし、docking-analysis-tools 側で `prepare_receptor()` を
呼ぶ際に `select_protein()` が非タンパク質残基を除去するかどうか確認してください。

**Q. 何フレームのトラジェクトリがあれば十分？**  
A. クラスター数の 10 倍以上のフレームが目安です（5 クラスター → 50 フレーム以上）。
ただし統計的な信頼性を得るためには 100〜500 フレーム程度が推奨です。

---

## 関連ドキュメント

- `notebooks/12_pose_clustering.ipynb` — クラスタリングの詳細な使い方
- `notebooks/14_convergence.ipynb` — トラジェクトリ収束確認
- [docking-analysis-tools Pipeline Integration](https://github.com/rkakamilan/docking-analysis-tools/blob/main/docs/pipeline_integration.md) — アンサンブルドッキング全体フロー
