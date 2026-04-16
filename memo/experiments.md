# 実験ガイド

本プロジェクトでは、合成データ生成および匿名化技術のプライバシーとユーティリティを評価する3種類の実験を提供する。

## データセット一覧

| データセット | パス | 行数 | 列数 | 説明 |
|---|---|---|---|---|
| Texas | `data/texas` | 100,000 | 18 | テキサス州入院患者記録 |
| Adult | `data/adult` | 32,561 | 15 | UCI Adult（国勢調査所得予測） |
| ML full | `data/movielens/ml32m_full` | 200,948 | 20 | MovieLens ml-32m（全ジャンル） |
| ML ncols_10 best | `data/movielens/ncols_10/eff_rank_best` | 200,948 | 10 | MovieLens（有効ランク上位10ジャンル） |
| ML ncols_10 worst | `data/movielens/ncols_10/eff_rank_worst` | 200,948 | 10 | MovieLens（有効ランク下位10ジャンル） |

各データセットは `.csv`（データ本体）と `.json`（カラムメタデータ）のペアで構成される。CLIの `-D` オプションには拡張子なしのパスを指定する。

MovieLens データセットの詳細（列構成・rarity 計算・前処理リポジトリへのリンク等）は `data/movielens/README.md` を参照。

---

## 実験1: Linkage（連結可能性攻撃）

### 概要

Membership Inference Attack (MIA) に基づき、特定のターゲットレコードが合成/匿名化データの元データに含まれていたかを攻撃者が推測できるかを評価する。Privacy Gain (PG = 1 - MIA Advantage) が高いほどプライバシー保護が強い。

### 評価フロー

1. 母集団からターゲットレコードを選択
2. 攻撃者の事前知識データ (rawA) をサンプリング
3. **攻撃訓練フェーズ**: 各モデル × 各ターゲットについてシャドウデータで MIA 分類器を訓練
   - Feature Set: `Naive`, `Histogram`, `Correlations`（合成モデル）、+ `Ensemble`（匿名化手法）
4. **評価フェーズ** (nIter回繰り返し): ターゲット有無の合成/匿名化データを生成し、攻撃者の推測精度を測定

### 実行コマンド

```bash
# 共通パターン
uv run python linkage_cli.py -D <データパス> -RC <run config> -O <出力先> [-W <並列数>]
```

#### Texas

```bash
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.texas.default.json -O tests/linkage
```

#### Adult

```bash
uv run python linkage_cli.py -D data/adult -RC tests/linkage/runconfig.adult.default.json -O tests/linkage
```

#### MovieLens（default: フルモデル構成）

```bash
uv run python linkage_cli.py -D data/movielens/ml32m_full              -RC tests/linkage/movielens/runconfig.ml32m_full.default.json              -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_best  -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_best.default.json  -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_worst -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_worst.default.json -O tests/linkage
```

#### MovieLens（lite: 軽量構成）

CTGAN, PATEGAN, IndependentHistogram を除外し、パラメータバリエーションを2つに削減した高速版。

```bash
uv run python linkage_cli.py -D data/movielens/ml32m_full              -RC tests/linkage/movielens/runconfig.ml32m_full.lite.json              -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_best  -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_best.lite.json  -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_worst -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_worst.lite.json -O tests/linkage
```

#### MovieLens の Targets 仕様

MovieLens 用 runconfig は `Targets` にデータセットごとの **rare user 5 人の固定 ID** を埋めてある（`rarity_count` が最大値のユーザー群からシード 42 でランダム抽出）。`nTargets=5` も合わせて指定しており、実際には **ランダム 5 + rare 固定 5 = 計 10 ターゲット** が評価される。rare ユーザー選定ロジックは `data/movielens/README.md` を参照。

### 出力

`ResultsMIA_<データセット名>.json` — ターゲットID × モデル × Run × FeatureSet ごとの攻撃結果

---

## 実験2: Inference（属性推論攻撃）

### 概要

合成/匿名化データから訓練した攻撃モデルが、ターゲットレコードの秘匿属性（Sensitive Attribute）を推測できるかを評価する。ターゲットが元データに含まれる場合 (IN) と含まれない場合 (OUT) の推測精度の差が Inference Advantage であり、Privacy Gain はその低減度を測る。

### 評価フロー

1. ターゲットレコードを選択
2. 各イテレーションで訓練データをサンプリング
3. **Raw攻撃**: 生データでの攻撃精度をベースラインとして計測
4. **モデル評価**: 各合成/匿名化モデルについて、ターゲット有無のデータから攻撃モデルを訓練し推測精度を測定
5. 数値属性は `LinReg`（線形回帰）、カテゴリ属性は `Classification`（ランダムフォレスト）で攻撃

### run config 固有パラメータ

| パラメータ | 説明 |
|---|---|
| `sensitiveAttributes` | 秘匿属性名と攻撃手法（`LinReg` / `Classification`）のマッピング |

### 実行コマンド

```bash
# 共通パターン
uv run python inference_cli.py -D <データパス> -RC <run config> -O <出力先> [-W <並列数>]
```

#### Texas

```bash
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.texas.default.json -O tests/inference
```

秘匿属性: `LENGTH_OF_STAY` (LinReg), `RACE` (Classification)

#### Adult

```bash
uv run python inference_cli.py -D data/adult -RC tests/inference/runconfig.adult.default.json -O tests/inference
```

秘匿属性: `hours-per-week` (LinReg), `race` (Classification)

#### MovieLens

MovieLens データセットには inference 用の run config は未作成。全カラムが同型（0-10のOrdinal）であり、秘匿属性の設定が自明でないため。

### 出力

`ResultsMLEAI_<データセット名>.json` — ターゲットID × 秘匿属性 × モデル × Run ごとの攻撃結果

---

## 実験3: Utility（ユーティリティ評価）

### 概要

合成/匿名化データで訓練した予測モデルの精度を、生データで訓練した場合と比較し、データユーティリティの劣化度を測定する。

### 評価フロー

1. データを train/test に分割（`dataFilter` で条件指定）
2. 各イテレーションで訓練データをサンプリング
3. **Raw評価**: 生データで予測モデルを訓練し、テストデータでの精度を計測
4. **モデル評価**: 各合成/匿名化モデルでデータを生成/変換し、そのデータで予測モデルを訓練して精度を計測
5. ターゲットの有無による個別レコード精度と、全テストデータでの集約精度を両方記録

### run config 固有パラメータ

| パラメータ | 説明 |
|---|---|
| `utilityTasks` | 予測タスクの定義（`RandForestClass` + 目的変数） |
| `dataFilter.train` | 訓練データのフィルタ条件（pandasのquery式） |
| `dataFilter.test` | テストデータのフィルタ条件 |
| `TestRecords` | 個別評価対象のテストレコードID |

### 実行コマンド

```bash
# 共通パターン
uv run python utility_cli.py -D <データパス> -RC <run config> -O <出力先> [-W <並列数>]
```

#### Texas

```bash
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.texas.default.json -O tests/utility
```

予測タスク: `RISK_MORTALITY`（ランダムフォレスト分類）、train/test分割: 2013年 / 2014年

#### Adult

```bash
uv run python utility_cli.py -D data/adult -RC tests/utility/runconfig.adult.default.json -O tests/utility
```

予測タスク: `label`（所得 <=50K / >50K のランダムフォレスト分類）、train/test分割: `age >= 0`（全データ）

#### MovieLens

MovieLens データセットには utility 用の run config は未作成。予測タスクの定義が必要。

### 出力

- `ResultsUtilTargets_<データセット名>.json` — 個別レコードの予測精度
- `ResultsUtilAgg_<データセット名>.json` — 集約予測精度

---

## 共通 run config パラメータ

全CLIで共通のパラメータ:

| パラメータ | 説明 | 典型値 |
|---|---|---|
| `nIter` | ゲーム繰り返し回数 | 5〜15 |
| `sizeRawT` | 各イテレーションの訓練データサイズ | 1000 |
| `sizeSynT` | 生成する合成データサイズ | 1000 |
| `nSynT` | 合成データ生成回数（評価用） | 5〜10 |
| `nTargets` | ランダム選択するターゲット数 | 0〜5 |
| `Targets` | 固定ターゲットのID一覧（`null`でランダムのみ） | - |

Linkage 固有:

| パラメータ | 説明 | 典型値 |
|---|---|---|
| `sizeRawA` | 攻撃者の事前知識データサイズ | 10000 |
| `nSynA` | シャドウ合成データ生成回数 | 10 |
| `nShadows` | シャドウモデル数 | 10 |

## モデル構成

### 合成データ生成モデル（generativeModels）

| モデル | パラメータ | 説明 |
|---|---|---|
| `IndependentHistogram` | `[bins]` | 独立ヒストグラム。bins=ビン数 |
| `BayesianNet` | `[bins, degree]` | ベイジアンネットワーク。bins=ビン数, degree=次数 |
| `PrivBayes` | `[bins, degree, epsilon]` | 差分プライバシー付きベイジアンネット。epsilon=プライバシー予算 |
| `CTGAN` | `[]` | Conditional GAN（デフォルトパラメータ） |
| `PATEGAN` | `[epsilon]` | PATE-GAN。epsilon=プライバシー予算 |

### 匿名化手法（sanitisationTechniques）

| 手法 | パラメータ | 説明 |
|---|---|---|
| `SanitiserNHS` | `[nbins, thresh_rare, max_quantile, k, drop_cols, quids]` | NHS式匿名化。k=匿名集合サイズ, quids=準識別子カラム |
| `SanitiserMondrian` | `[k, quids, drop_cols]` | Mondrian k-匿名化。k=匿名集合サイズ, quids=準識別子カラム |

### default vs lite（MovieLens）

| | default | lite |
|---|---|---|
| 合成モデル | IndependentHistogram(2), BayesianNet(4), PrivBayes(3), CTGAN(1), PATEGAN(2) = 12種 | BayesianNet(2), PrivBayes(2) = 4種 |
| SanitiserNHS | k=2,5,10,25 | k=5,25 |
| SanitiserMondrian | k=2,5,10,25 | k=5,25 |

---

## 結果の可視化

### ノートブック一覧

| ノートブック | 内容 |
|---|---|
| `notebooks/Analyse Results.ipynb` | Texas データセットの Linkage / Inference / Utility 結果分析 |
| `notebooks/Privacy-Utility tradeoff.ipynb` | Texas の Privacy-Utility トレードオフプロット |
| `notebooks/Privacy-Utility tradeoff (adult).ipynb` | Adult の Privacy-Utility トレードオフプロット |
| `notebooks/Privacy Gain (MovieLens).ipynb` | MovieLens の Linkage プライバシーゲイン可視化 |

### 分析ユーティリティ

- `utils/analyse_results.py` — 結果JSONの読み込み・集約関数（`load_results_linkage`, `load_results_inference`, `load_results_utility`）
- `utils/tradeoff_plot.py` — Privacy-Utility トレードオフの可視化

---

## 実行例: 全実験の一括実行（Texas）

```bash
# Linkage
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.texas.default.json -O tests/linkage

# Inference
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.texas.default.json -O tests/inference

# Utility
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.texas.default.json -O tests/utility
```

## 実行例: 全実験の一括実行（Adult）

```bash
# Linkage
uv run python linkage_cli.py -D data/adult -RC tests/linkage/runconfig.adult.default.json -O tests/linkage

# Inference
uv run python inference_cli.py -D data/adult -RC tests/inference/runconfig.adult.default.json -O tests/inference

# Utility
uv run python utility_cli.py -D data/adult -RC tests/utility/runconfig.adult.default.json -O tests/utility
```

## 実行例: MovieLens（Linkageのみ）

```bash
# lite版で3データセット（ml32m_full + ncols_10 best/worst）実行
uv run python linkage_cli.py -D data/movielens/ml32m_full              -RC tests/linkage/movielens/runconfig.ml32m_full.lite.json              -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_best  -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_best.lite.json  -O tests/linkage
uv run python linkage_cli.py -D data/movielens/ncols_10/eff_rank_worst -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_worst.lite.json -O tests/linkage
```
