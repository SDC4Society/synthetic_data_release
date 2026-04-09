# SanitiserMondrian 追加設計

## 背景と目的

現在のプライバシー評価パイプラインでは、伝統的匿名化手法として `SanitiserNHS`（NHS England 流の一般化＋希少カテゴリ削除＋k-匿名性）のみが評価対象となっている。これに加えて、k-匿名化の代表的アルゴリズムである **Mondrian**（LeFevre et al. 2006）を評価対象として追加し、NHS との比較を可能にする。

実装には既に `pyproject.toml` で依存登録済みの `k-anonymization` ライブラリ（`k_anonymization.algorithms.local_recoding.mondrian.ClassicMondrian`）を使用する。

## スコープ

- `SanitiserMondrian` クラスを追加し、既存の `SanitiserNHS` と同じ runconfig 経由で利用可能にする
- linkage / inference / utility の3つの評価 CLI で動作させる
- 攻撃器・feature extractor・predictive model の既存コードには手を入れない

**スコープ外**:
- `k_anonymization` ライブラリ自体の拡張
- 他の k-匿名化アルゴリズム（kmember, oka 等）の追加
- Mondrian の `SUMMARIZATION` / `GENERALIZATION` 戦略のサポート（MEAN_MODE 固定）

## 前提となる発見事項

調査の結果、以下が確認できている:

1. **`data/texas.json` の QID 候補列は全てカテゴリ/序数列**。例えば `PAT_AGE` は `type: Ordinal`, `i2s: ["00", "01", ..., "22"]` で、CSV 上も `"01"`, `"16"` のようなゼロ埋め文字列。`bins` は存在しない。
2. NHS が `pandas.cut` で数値 quid をビニングする経路は、texas データでは実際には発火していない。
3. `ClassicMondrian` は `MEAN_MODE` 戦略下でカテゴリ列に対してグループ最頻値を返す。この値は必ず元カテゴリ集合 (`i2s`) 内に収まる。
4. 従って **Mondrian の出力は既存 metadata のスキーマ契約を自動的に満たす** 。出力 metadata の動的生成は不要。
5. ただし `k_anonymization.Dataset` は props.json をディスクから読む前提のため、インメモリ用のアダプタが必要。
6. `LocalRecodingAlgorithm.__init__` は `get_max_ranges` 経由で `hierarchies[idx].height` を要求するため、カテゴリ QID には最低限の hierarchy オブジェクトが必要。

## 設計

### アーキテクチャ

```
runconfig
  └─ "SanitiserMondrian": [[k, quids, drop_cols], ...]
      │
      ▼
MODEL_REGISTRY (utils/parallel.py)
  └─ create_model(cfg, metadata) → SanitiserMondrian(metadata, k, quids, drop_cols)
      │
      ▼
CLI worker (linkage/inference/utility)
  │  1. model = create_model(cfg, metadata)
  │  2. attack_metadata = model.get_output_metadata(metadata)  ← 恒等
  │  3. san = model.sanitise(data)
  │  4. feature extractor / attack / predictive model に attack_metadata を渡す
  ▼
SanitiserMondrian.sanitise(data)
  │  1. drop_cols を除外
  │  2. NaN imputation (SimpleImputer)
  │  3. _InMemoryDataset を組み立てる
  │     - df:             imputed DataFrame (ITableDF でラップ)
  │     - qids:           self.quids
  │     - qids_idx:       列インデックス
  │     - is_categorical: 全て True (texas の quid は全てカテゴリ)
  │     - hierarchies:    各 QID に深さ1の _FlatHierarchy を割当
  │  4. ClassicMondrian(dataset, k, MEAN_MODE).anonymize()
  │  5. 結果 DataFrame を元の data.index で返す
```

### コンポーネント

#### 1. `sanitisation_techniques/sanitiser.py` の拡張

基底 `Sanitiser` クラスに以下のメソッドを追加:

```python
def get_output_metadata(self, input_metadata):
    """Return metadata describing the sanitiser output schema.

    Default implementation returns input_metadata unchanged, meaning
    the sanitiser preserves the input schema. Subclasses may override
    if they transform the schema (e.g., quantize numeric QIDs).
    """
    return input_metadata
```

`SanitiserNHS` も暗黙的にこのデフォルトを継承するのみ（書き換えなし）。

#### 2. `sanitisation_techniques/sanitiser_mondrian.py`（新規）

```python
from pandas import DataFrame
from sklearn.impute import SimpleImputer

from k_anonymization.algorithms.local_recoding.mondrian import ClassicMondrian
from k_anonymization.algorithms.local_recoding.local_recoding_algorithm import (
    GroupAnonymizationBuiltIn,
)
from k_anonymization.core.frame import ITableDF

from sanitisation_techniques.sanitiser import Sanitiser
from utils.constants import CATEGORICAL, ORDINAL, NUMERICAL


class _FlatHierarchy:
    """Minimal hierarchy stub satisfying ClassicMondrian's attribute access.

    ClassicMondrian + MEAN_MODE only touches `hierarchies[idx].height`
    via get_max_ranges. A flat hierarchy of depth 1 is sufficient.
    """
    height = 1


class _InMemoryDataset:
    """Duck-typed substitute for k_anonymization.core.Dataset.

    Provides only the attributes that ClassicMondrian and
    LocalRecodingAlgorithm.__init__ access, without the
    props.json/CSV disk loading of the real Dataset class.
    """
    def __init__(self, df, qids, qids_idx, is_categorical, hierarchies):
        self.df = ITableDF(df)
        self.qids = qids
        self.qids_idx = qids_idx
        self.is_categorical = is_categorical
        self.hierarchies = hierarchies
        self.target = None


class SanitiserMondrian(Sanitiser):
    """Mondrian k-anonymization sanitiser (MEAN_MODE strategy)."""

    def __init__(self, metadata, k=5, quids=None, drop_cols=None):
        if not quids:
            raise ValueError("SanitiserMondrian requires at least one QID")
        self.metadata = metadata
        self.k = int(k)
        self.quids = list(quids)
        self.drop_cols = list(drop_cols) if drop_cols else []
        self.datatype = DataFrame

        self.ImputerCat = SimpleImputer(strategy="most_frequent")
        self.ImputerNum = SimpleImputer(strategy="median")

        self.__name__ = f"SanitiserMondrianK{self.k}"
        self.trained = False

    def sanitise(self, data):
        # 1. drop unwanted columns
        work = data.drop(columns=self.drop_cols, errors="ignore").copy()
        original_index = work.index
        original_columns = list(work.columns)

        missing = [q for q in self.quids if q not in work.columns]
        if missing:
            raise ValueError(f"QIDs not found in data: {missing}")

        # 2. NaN imputation by column type
        work = self._impute(work)

        # 3. build in-memory dataset
        qids_idx = [original_columns.index(q) for q in self.quids]
        is_categorical = [True] * len(self.quids)  # texas quids are all categorical
        hierarchies = {idx: _FlatHierarchy() for idx in qids_idx}
        dataset = _InMemoryDataset(
            df=work.reset_index(drop=True),
            qids=self.quids,
            qids_idx=qids_idx,
            is_categorical=is_categorical,
            hierarchies=hierarchies,
        )

        # 4. run Mondrian
        algo = ClassicMondrian(
            dataset=dataset,
            k=self.k,
            group_anonymization=GroupAnonymizationBuiltIn.MEAN_MODE,
        )
        algo.anonymize()

        # 5. restore index
        anon = DataFrame(algo.anon_data, columns=original_columns)
        anon.index = original_index[: len(anon)]
        return anon

    def _impute(self, df):
        cat_cols, num_cols = [], []
        for cdict in self.metadata["columns"]:
            col = cdict["name"]
            if col not in df.columns:
                continue
            if cdict["type"] in [CATEGORICAL, ORDINAL]:
                cat_cols.append(col)
            elif cdict["type"] in NUMERICAL:
                num_cols.append(col)
        if cat_cols:
            df[cat_cols] = self.ImputerCat.fit_transform(df[cat_cols])
        if num_cols:
            df[num_cols] = self.ImputerNum.fit_transform(df[num_cols])
        return df
```

**注**: `get_output_metadata` はオーバーライド不要（基底の恒等実装で十分）。

#### 3. `utils/parallel.py` へのレジストリ登録

```python
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian

MODEL_REGISTRY = {
    ...,
    "SanitiserMondrian": SanitiserMondrian,
}
```

#### 4. CLI worker 関数の修正

以下5つの worker 関数に1行ずつ追加:

- `linkage_cli.py`: `linkage_attack_worker`, `linkage_eval_worker`
- `inference_cli.py`: `inference_attack_worker`, `inference_eval_worker`
- `utility_cli.py`: `utility_eval_worker`

パターン:

```python
model = create_model(model_config, metadata)
model.multiprocess = False
# ↓ 追加
attack_metadata = (
    metadata if is_generative_model(model)
    else model.get_output_metadata(metadata)
)
# 以降、feature extractor / attack / predictive model の metadata 引数を
# attack_metadata に差し替える
```

現在の恒等実装下では挙動は何も変わらないが、将来の拡張（別 sanitiser で本当に metadata を書き換えたい場合）に備えて配線を先に入れておく。

#### 5. runconfig 更新

`tests/{linkage,inference,utility}/runconfig.default.json` の `sanitisationTechniques` に `SanitiserMondrian` エントリを追加:

```json
"SanitiserMondrian": [
  [2,  ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
  [5,  ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
  [10, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
  [25, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []]
]
```

位置引数の順序は `[k, quids, drop_cols]`。`_deep_tuple` による tuple 変換は既存配線で処理済み。

### データフロー（quid = `PAT_AGE` の例）

```
Raw CSV: PAT_AGE = "16"  (Ordinal, "00"〜"22")
  │
  ▼ SanitiserMondrian.sanitise
  │  factorize → 整数コード
  │  Mondrian: メディアン分割（コード順、これは Ordinal の元順とは限らない）
  │  MEAN_MODE: グループ内最頻の元文字列 (例: "15")
  ▼
Anonymized DataFrame: PAT_AGE = "15"  (必ず元 i2s 集合内)
  │
  ▼ HistogramFeatureSet
  │  CategoricalDtype(categories=["00"〜"22"])
  │  value_counts → 特徴量ベクトル
  ▼
MIA / AIA 分類器入力
```

### エラーハンドリング

| 状況 | 挙動 |
|---|---|
| `quids` が空/None | `ValueError` |
| `quids` に存在しない列名 | `ValueError` |
| `k > len(data)` | 分割ゼロで全データ1グループ（許容、NHSと同等） |
| quid のユニーク値が1つのみ | Mondrian が自然に分割を諦める（問題なし） |
| `ClassicMondrian` 内部例外 | 呼び出し元に伝播 |

### テスト計画

#### 単体テスト (`tests/test_sanitisation.py`)

- `SanitiserMondrian(metadata, k=3, quids=[...])` で germancredit_test を sanitise
- 出力の行数 ≤ 入力、列数・index が保持される
- 各 quid 列の値が元 `i2s` 集合に含まれる
- 各グループ（quid の組み合わせ）のサイズが k 以上
- `quids=None` で `ValueError`
- 存在しない quid で `ValueError`

#### 統合テスト (`tests/test_parallel.py`)

- `create_model(("SanitiserMondrian", 3, ["age", "sex"], []), metadata)` でインスタンス化成功
- `MODEL_REGISTRY` から取れる

#### エンドツーエンド（手動）

- 小さい `nIter` で `linkage_cli.py -D data/texas -RC tests/linkage/runconfig.default.json -O tests/linkage` を完走
- `inference_cli.py`, `utility_cli.py` も同様

## リスクと検証項目

### 実装初期に検証すべき3点

1. **`Hierarchy` 依存の副作用回避**
   `_FlatHierarchy` ダミーが `get_max_ranges` で正しく動くか（`.height` のみアクセスされることは確認済みだが実機確認）。もし他属性にアクセスが及んだ場合は、実際の `k_anonymization.core.hierarchy.Hierarchy` を使った平坦階層を生成する方式にフォールバック。ただしその場合 `ipywidgets` が worker プロセスで import される副作用が発生する。

2. **`_construct_anon_data` 出力形式**
   `ClassicMondrian.anonymize()` 完了後の `algo.anon_data` が `ITableDF`（= `pandas.DataFrame` サブクラス）であることを前提に `DataFrame(...)` 経由で再構築する。型が想定外の場合は `algo.anon_data.values` から組み直す。

3. **`__ID` 列と index 復元**
   `LocalRecodingAlgorithm.anonymize` は内部で `__ID` 列を追加→並び戻し→pop する。結果は元の row order を保つが pandas index はリセットされる可能性がある。`sanitise` の末尾で `anon.index = original_index[:len(anon)]` で元 index を復元する。NHS の挙動（グループ削除によって index が欠ける）と整合。

### パイプライン不変条件チェック

| 不変条件 | 維持されるか |
|---|---|
| sanitise 出力が元 metadata の `i2s` カテゴリ集合に収まる | ✓ (MEAN_MODE はカテゴリ列で最頻値=元集合内) |
| 出力 DataFrame の列名・列順が保持される | ✓ (`original_columns` で復元) |
| 並列 worker で pickle 可能 | ✓ (`SanitiserMondrian` は pure Python) |
| `multiprocess = False` 設定が効く | ✓ (既存 Sanitiser と同じ扱い) |
| runconfig の list → tuple 変換 (`_deep_tuple`) が効く | ✓ (既存配線のまま) |

## 実装の順序

1. `Sanitiser` 基底クラスに `get_output_metadata` 追加
2. `_FlatHierarchy` + `_InMemoryDataset` + `SanitiserMondrian` を最小実装
3. REPL/単体テストで germancredit_test に対する sanitise 動作確認（`_FlatHierarchy` の妥当性検証を含む）
4. `MODEL_REGISTRY` 登録
5. 3 CLI worker に `get_output_metadata` 呼び出し配線
6. 単体テスト + 統合テスト追加
7. runconfig 更新
8. 3 CLI のエンドツーエンド手動確認

## 非目標と将来の拡張余地

- **数値 quid のサポート**: 現在の texas データでは不要だが、将来データが変わった場合に備えて `is_categorical` 判定を metadata の type ベースに切り替える拡張は容易
- **Mondrian 以外の戦略**: `SUMMARIZATION` / `GENERALIZATION` はパイプライン契約（元カテゴリ集合内）を破るため、採用する場合は `get_output_metadata` のオーバーライドが必要
- **他の k-匿名化アルゴリズム**: 同じ `_InMemoryDataset` アダプタを流用して `kmember`, `oka` 等を追加可能
