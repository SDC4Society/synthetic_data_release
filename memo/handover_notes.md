# 引き継ぎメモ

最終更新: 2026-04-09

現在、Texas / Adult / MovieLens の各データセットで実験を実行し結果を得られる状態になっている。以下、実装の経緯と今後の改善ポイントをまとめる。

---

## 1. SanitiserMondrian（ClassicMondrian）の実装状況

Chanhさんのライブラリ（`SDC4Society/k_anonymization`）から ClassicMondrian を MEAN_MODE 戦略で統合し、`SanitiserMondrian` として実装した（`sanitisation_techniques/sanitiser_mondrian.py`）。Texas データセットで Privacy-Utility トレードオフの曲線をプロットできるところまで動作している。

### 改善ポイント: Accuracy の調整

Texas データセットで SanitiserMondrian の Accuracy が他の手法より低めに出ている。以下のチューニングで改善する余地がある:

- **パラメータ調整**: `k` の値やQIDの組み合わせを変えて試す。現在のQIDは `PAT_STATE, SEX_CODE, RACE, ETHNICITY, PAT_AGE` の5列、`k=2,5,10,25`。
- **型変換の影響**: MEAN_MODE で Integer 型QIDが Float に変換される。`get_output_metadata` で反映済みだが、下流タスクでの影響は要確認。
- **前処理の追加**: 外れ値除去を入れると MEAN_MODE の汎化結果が安定する可能性がある。
- **カテゴリ階層**: 現在 `_FlatHierarchy`（`height=1`）のスタブで代替しており、適切な階層を定義すればカテゴリ属性の汎化品質が向上するかもしれない。

---

## 2. Adult データセットの取得方法

フォーク元のコードでは Adult データセットを S3 (`sdgym.s3.amazonaws.com`) から `.npz` + `.json` 形式で取得する想定だったが、S3バケットにアクセスできなかったため代替手段で取得した。

### 実施内容

- UCI Machine Learning Repository から Adult データセット（`adult.data`）を直接取得し `data/adult.csv` に変換
- メタデータ `data/adult.json` を手動作成

### メタデータ作成時の判断

- カラム型: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` → `Integer`、`education` → `Ordinal`（学歴順）、その他 → `Categorical`
- `education` の `i2s` 順序: `Preschool` → `Doctorate` の学歴順で配置
- `age` の `bins`: `[16, 25, 35, 45, 55, 65, 75, 91]`（BayesianNet等で使用）
- `min` / `max` は実データから算出、`i2s` は実データのユニーク値を列挙（欠損値 `?` を含む）

フォーク元がS3で提供していたデータと前処理（欠損値の扱い等）が異なる可能性はある。結果に明らかな異常がなければ実用上は問題ないと思われる。

---

## 3. MovieLens データセットの状況

MovieLens データセット（`ml_col3_*`, `ml_col19_*`）は全カラムが 0-10 の Ordinal 型（映画ジャンル別レーティング）。

- **Linkage（MIA）**: 実装済み。プライバシーゲインの算出・可視化ノートブックまで完了。run config は default と lite の2種類。
- **Inference / Utility**: 未実装。全カラムが同型レーティング値のため、`sensitiveAttributes` や `utilityTasks` の目的変数をどう設定するかの設計が必要。例えば「特定ジャンルの評価を他ジャンルから予測」といったタスク定義が考えられる。

---

## 4. SanitiserNHS と高次元データ

SanitiserNHS を MovieLens col19（全19列をQIDに指定、`k=25`）で実行すると、k-匿名性を満たさないグループの抑制（レコード削除）が連鎖し、最終的に全行または大部分の行が削除される現象が起きる。

これは SanitiserNHS の仕様上の制約で、QID数が多いと組み合わせ空間が爆発し、各グループのサイズが `k` 未満になりやすいため。19列×11値（0-10）の組み合わせ空間に対して `sizeRawT=1000` では密度が極めて低い。

col3（3列）では問題なく動作する。col19 で SanitiserNHS を使う場合は、QIDを全列ではなく一部に限定するか、`k` を小さくする必要がある。現在の default/lite run config では全列をQIDに指定しているので、col19 で SanitiserNHS の結果が空になる可能性がある点に留意。

---

## 5. 4/9 作業分のコードについて

4/9の作業は Claude Code による実装支援を多用した。動作確認済みだが、以下の変更箇所はレビューが手薄なので、不審な挙動があればまずこのあたりを確認するとよい。

- **`sanitiser_nhs.py:147-152`** — `cat_cols` / `num_cols` が空の場合の Imputer スキップ処理（MovieLens のような全Ordinalデータ対応）
- **`sanitiser_nhs.py:77`** — ハイフン入りカラム名のpandas queryエスケープ（バッククォート追加）
- **`reconstruction.py:276-283`** — `_get_proba` メソッド追加。`predict_proba` のクラスインデックスの不一致を修正
- **`sanitiser_mondrian.py:117-123`** — `get_output_metadata` で Integer→Float の型変換を反映
- **`feature_sets/model_agnostic.py`, `independent_histograms.py`, `bayes.py`** — 全 FeatureSet の `extract()` に空 DataFrame ガードを追加。SanitiserNHS が全行抑制した場合にゼロベクトルを返す処理。ゼロベクトルが MIA 分類器の学習・推論に与える影響は未評価。
- **`notebooks/Privacy Gain (MovieLens).ipynb`** — `load_results_linkage` のDataset名パース問題を回避する独自読み込み関数（本来は `utils/analyse_results.py` 側を修正するのが望ましい）
- **MovieLens用 run config（8ファイル）** — パラメータは暫定値。実験結果を見ながら調整していく想定

---

## 6. AIA（Attribute Inference Attack）の結果について

以前から確認している点として、Inference（AIA）の Privacy Gain が元論文（Stadler et al. 2020 "Groundhog Day"）の傾向と異なり、最大でも 0 付近にとどまっている。

原因はまだ絞り込めていないが、候補としては:

- `inference_cli.py` の Raw 攻撃とモデル攻撃の比較ロジック
- `reconstruction.py` の `get_likelihood` / `predict_proba` 周り（4/9に一部修正済み）
- IN/OUT 判定と攻撃精度の計算方法の定義差異
- `sensitiveAttributes` のタスク種別（`LinReg` vs `Classification`）の影響

実験自体は正常に完走し結果は得られているので、余裕のあるときに論文の実装と突き合わせて調査するとよい。

---

## 7. トレードオフ曲線のチューニング

現在のプロットでは、プライバシーパラメータ（ε, k, bins）に対して Privacy Gain や Accuracy が綺麗な単調曲線にならないケースがある。これは主に試行回数とパラメータグリッドの問題と考えられる。

### 現在の実験パラメータ

| パラメータ | Texas / Adult | MovieLens |
|---|---|---|
| `nIter` | 15 | 5 |
| `nShadows` (linkage) | 10 | 10 |
| `nSynA` (linkage) | 10 | 10 |
| `nSynT` | 5 (linkage) / 10 (inference, utility) | 5 |
| `sizeRawT` | 1000 | 1000 |

### 改善の方向性

- **`nIter` の増加**: 15→50〜100 程度に増やすと平均が安定し、曲線が滑らかになるはず。ただし実行時間とのトレードオフ。
- **パラメータグリッドの追加**: PrivBayes の `ε` は現在 `[0.1, 1.0, 10.0]` の3点のみ。中間値（0.5, 2.0, 5.0 等）を足すとプロットの解像度が上がる。
- **`sizeRawT` の増加**: 各イテレーションのサンプル数を増やすとモデル学習のばらつきが減る。

---

## 8. その他

### 環境・依存関係

- Python 3.9+ / uv で管理。`.venv` はリポジトリルートに存在
- CTGAN: フォーク版 `SDC4Society/CTGAN`（`pyproject.toml` にgit依存として記載）
- k-anonymization: フォーク版 `SDC4Society/k_anonymization`
- git依存パッケージの更新時は `uv cache clean` を実行すること（キャッシュが古いコミットを参照する問題がある）

### リポジトリ構造メモ

- `tests/linkage/` 等に結果JSON（`ResultsMIA_*.json`等）が出力される。`.gitignore` に入っていないので、大きなファイルのコミットに注意
- `tests/` がユニットテストと実験結果出力先を兼ねている。将来的に分離を検討

### `load_results_linkage` のDataset名パース

`utils/analyse_results.py` の `load_results_linkage` はファイル名を `_` で分割した最後の要素を Dataset 名とする。`ResultsMIA_ml_col3_eff_rank_best.json` → `best` になり、MovieLens のデータセット名が区別できない。MovieLens ノートブックでは独自読み込み関数で回避済みだが、`analyse_results.py` 側の修正が望ましい。

### 並列実行

- 全CLIが `multiprocessing.Pool` で並列実行。`-W 1` でシングルプロセス実行可能（デバッグ時に便利）
- PATEGAN（TensorFlow）はワーカープロセス内の GPU メモリに注意

### ドキュメント一覧

- `memo/experiments.md` — 全実験の実行方法・パラメータ解説
- `memo/adding_sdg_method.md` — 新規SDG手法の追加手順
- `memo/dataset_format.md` — データセットフォーマット
- `CLAUDE.md` — Claude Code 向けプロジェクト概要
