# SDG手法の追加方法

新しい合成データ生成（SDG）手法をフレームワークに追加する手順。

---

## 1. GenerativeModel を継承したクラスを作成

`generative_models/` ディレクトリに新しいファイルを作成し、`GenerativeModel` 基底クラスを継承する。

```python
# generative_models/my_model.py
from pandas import DataFrame
from generative_models.generative_model import GenerativeModel
from utils.logging import LOGGER


class MyModel(GenerativeModel):

    def __init__(self, metadata, param1=10, param2=1.0, multiprocess=True):
        """
        :param metadata: dict: データセットのメタデータ（columns情報を含む）
        :param param1: モデル固有のハイパーパラメータ
        :param param2: モデル固有のハイパーパラメータ
        :param multiprocess: bool: 並列実行の可否（Poolワーカー内ではFalseに設定される）
        """
        self.metadata = metadata
        self.datatype = DataFrame       # 必須: 入出力の型
        self.multiprocess = bool(multiprocess)  # 必須: 並列制御フラグ
        self.trained = False            # 必須: 学習済みフラグ
        self.__name__ = f'MyModelP{param1}'  # 必須: 結果JSONのキー名になる

        # モデル固有の初期化
        self.param1 = param1
        self.param2 = param2

    def fit(self, data):
        """
        データに対してモデルを学習する。
        :param data: DataFrame: 学習データ（IDインデックス付き）
        """
        assert isinstance(data, self.datatype)
        LOGGER.debug(f'Start fitting {self.__name__} to data of shape {data.shape}...')

        # ここにモデルの学習ロジックを実装
        # ...

        self.trained = True

    def generate_samples(self, nsamples):
        """
        学習済みモデルから合成データを生成する。
        :param nsamples: int: 生成するサンプル数
        :return: DataFrame: 合成データ（元データと同じカラム構成）
        """
        assert self.trained, "Model must be fitted to some data first"
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        # ここに合成データ生成ロジックを実装
        # ...

        return synthetic_data  # DataFrame を返す
```

### 必須属性

| 属性 | 型 | 説明 |
|---|---|---|
| `self.datatype` | type | `DataFrame` を設定。入力データの型チェックに使用される |
| `self.__name__` | str | 結果JSONのキー名。パラメータを含めてユニークにする（例: `MyModelP10`） |
| `self.multiprocess` | bool | `True` でPoolワーカーから呼ばれた際に自動で `False` に設定される |
| `self.trained` | bool | `fit()` 完了後に `True` に設定。`generate_samples()` で事前チェックに使用 |
| `self.metadata` | dict | コンストラクタで受け取ったメタデータ。内部処理で参照 |

### `__init__` のシグネチャ規則

コンストラクタの第1引数は必ず `metadata`。それ以降のパラメータが run config の `generativeModels` から位置引数として渡される。

```python
# run config での指定
"MyModel": [[10, 1.0], [25, 2.0]]

# → MyModel(metadata, 10, 1.0) と MyModel(metadata, 25, 2.0) が生成される
```

### `fit()` の入力データ

- `DataFrame` 形式（`ID0`, `ID1`, ... のインデックス付き）
- カラム名はメタデータの `columns[].name` と一致
- Categorical/Ordinal カラムは文字列型（`object`）
- Integer/Float カラムは数値型

### `generate_samples()` の出力データ

- 入力と同じカラム構成の `DataFrame` を返す
- Categorical/Ordinal カラムは元のカテゴリ値（文字列）で出力
- インデックスは任意（評価側でリセットされる）

---

## 2. MODEL_REGISTRY に登録

`utils/parallel.py` の `MODEL_REGISTRY` にクラスを追加する。

```python
# utils/parallel.py
from generative_models.my_model import MyModel

MODEL_REGISTRY = {
    "IndependentHistogram": IndependentHistogram,
    "BayesianNet": BayesianNet,
    "PrivBayes": PrivBayes,
    "CTGAN": CTGAN,
    "PATEGAN": PATEGAN,
    "SanitiserNHS": SanitiserNHS,
    "SanitiserMondrian": SanitiserMondrian,
    "MyModel": MyModel,  # ← 追加
}
```

ここで登録したキー名が run config の `generativeModels` で使用する名前になる。

---

## 3. run config に追加

run config JSON の `generativeModels` セクションに追加する。

```json
{
  "generativeModels": {
    "MyModel": [[10, 1.0], [25, 2.0]]
  }
}
```

各リストが `__init__(metadata, *params)` の `*params` として展開される。パラメータなしの場合は `[[]]` とする（CTGANの例）。

---

## 4. 動作確認

### 単体テスト

```python
# tests/test_my_model.py
import unittest
from utils.datagen import load_local_data_as_df

class TestMyModel(unittest.TestCase):
    def test_fit_generate(self):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df.sample(500)

        from generative_models.my_model import MyModel
        model = MyModel(metadata, param1=10)
        model.fit(sample)

        syn = model.generate_samples(100)
        self.assertEqual(len(syn), 100)
        self.assertEqual(list(syn.columns), list(sample.columns))
```

```bash
uv run python -m unittest tests/test_my_model.py
```

### Linkage CLI での統合テスト

```bash
uv run python linkage_cli.py -D data/adult -RC tests/linkage/runconfig_my_model.json -O tests/linkage -W 1
```

`-W 1` でシングルプロセス実行するとデバッグが容易。

---

## 既存実装の参考例

| 実装 | ファイル | 特徴 |
|---|---|---|
| `IndependentHistogram` | `generative_models/data_synthesiser.py` | 最もシンプル。独立ヒストグラムからサンプリング |
| `BayesianNet` | `generative_models/data_synthesiser.py` | Greedy Bayesでネットワーク構築。`_read_meta` でメタデータ変換 |
| `PrivBayes` | `generative_models/data_synthesiser.py` | `BayesianNet` を継承し差分プライバシーを追加 |
| `CTGAN` | `generative_models/ctgan.py` | 外部ライブラリ (`ctgan`) をラップ。最小限の実装例 |
| `PATEGAN` | `generative_models/pate_gan.py` | TensorFlowベース。メタデータの独自パース例 |

外部ライブラリをラップする場合は `CTGAN` の実装（50行程度）が最も参考になる。

---

## 注意事項

- **並列実行**: `linkage_cli.py` / `inference_cli.py` はモデルをPoolワーカー内で実行する。ワーカー内では `model.multiprocess = False` が設定されるため、モデル内部で子プロセスを生成しないこと。
- **pickle互換性**: `multiprocessing.Pool` でモデルの config タプルをワーカーに送り `create_model()` で再構築するため、`__init__` の引数だけでモデルが再現可能であること。モデルインスタンス自体はpicklされない。
- **`__name__` の一意性**: 同じモデルでパラメータが異なる場合、`__name__` が異なる値を返す必要がある。結果の集約キーとして使われるため、衝突すると結果が上書きされる。
- **メタデータの利用**: `metadata['columns']` にカラム定義がある。型は `Categorical`, `Ordinal`, `Integer`, `Float` の4種。`metadata['categorical_columns']` 等にはインデックスリストが入る。
