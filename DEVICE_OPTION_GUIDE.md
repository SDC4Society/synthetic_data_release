# GPU/CUDA --device オプション使用ガイド

## 概要

すべての CLI スクリプト（`linkage_cli.py`, `inference_cli.py`, `utility_cli.py`）に `--device` オプションが追加されました。このオプションにより、CPU または GPU でプライバシー評価を実行できます。

## 使用方法

### CPU での実行

```bash
# Linkage privacy evaluation
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cpu

# Attribute inference evaluation  
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cpu

# Utility evaluation
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility --device cpu
```

### GPU での実行

```bash
# Linkage privacy evaluation
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cuda:0

# Attribute inference evaluation
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cuda:0

# Utility evaluation
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility --device cuda:0
```

### デフォルト動作

`--device` オプションを指定しない場合：
- NVIDIA GPU が利用可能 → `cuda:0` を自動使用
- GPU が利用不可 → `cpu` を使用

```bash
# 自動デバイス選択
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference
```

## オプション値

- `cpu`: CPU のみを使用
- `cuda:0`: GPU (CUDA device 0) を使用
- `cuda:1`: GPU (CUDA device 1) を使用 (複数GPU環境)

## 環境変数による設定

CLI オプション以外に、環境変数 `SYNTHETIC_DATA_DEVICE` でもグローバルデバイスを指定できます：

```bash
# 環境変数で GPU を指定
export SYNTHETIC_DATA_DEVICE=cuda:0
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference

# または両方指定（CLI オプションが優先）
export SYNTHETIC_DATA_DEVICE=cpu --device cuda:0  # cuda:0 が使用されます
```

## 実装の詳細

### CLI レベルの機構

1. `--device` オプションの解析
2. `SYNTHETIC_DATA_DEVICE` 環境変数の設定
3. CPU 指定時は `CUDA_VISIBLE_DEVICES=''` を設定（TensorFlow CUDA エラー回避）

### モデルレベルの機構

各モデルは初期化時に `device` パラメータを受け入れます：

```python
from generative_models.aim import AIM
from generative_models.gem import GEM
from generative_models.tabddpm import TabDDPM

# GPU での実行
model_aim = AIM(metadata, epsilon=1.0, device='cuda:0')
model_gem = GEM(metadata, epsilon=1.0, device='cuda:0')
model_tabddpm = TabDDPM(metadata, device='cuda:0')

# CPU での実行
model_aim_cpu = AIM(metadata, epsilon=1.0, device='cpu')
```

## 対応モデル

以下のモデルで GPU/CPU デバイス指定が全サポートされています：

| モデル | PyTorch | TensorFlow | JAX |
|--------|---------|------------|-----|
| AIM | ✓ | - | - |
| GEM | ✓ | - | - |
| TabDDPM | ✓ | - | - |
| DP_MERF | ✓ | - | - |
| CTGAN | ✓ | - | - |
| PATEGAN | - | ✓ | - |
| PrivateGSD | - | - | ✓ |
| RAPpp | - | - | ✓ |
| IndependentHistogram | - | - | - |
| BayesianNet | - | - | - |
| PrivBayes | - | - | - |
| PrivMRF | ✓ | - | - |
| PrivSyn | - | - | - |

## パフォーマンスに関する注意

- **GPU メモリ**: 大規模データセットの場合、GPU メモリが不足することがあります。その場合は `--device cpu` を使用するか、バッチサイズを減らしてください。
- **CPU**: 小規模実験や資源制約のある環境では CPU での実行が適切です。
- **複数 GPU**: 複数 GPU 環境では `--device cuda:0`, `cuda:1` などで GPU を指定できます。

## トラブルシューティング

### CUDA エラーが発生する場合

```bash
# --device cpu を明示的に指定
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cpu
```

### グローバルなデバイス設定の優先順位

1. CLI `--device` オプション（最優先）
2. 環境変数 `SYNTHETIC_DATA_DEVICE`
3. 自動検出（GPU 優先、GPU なし → CPU）

## テスト

デバイスオプションが正常に機能しているか確認：

```bash
# テストスクリプトを実行
python test_device_cli_option.py
```

期待される出力：
- ✓ CPU device option: Environment variable set and CUDA cleared
- ✓ GPU device option: Device passed through correctly
