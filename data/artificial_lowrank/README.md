# synthetic_lowrank データセット

`Mondorian_vs_effective_rank` リポジトリで用いている
`generate_discrete_lowrank` と同一の手順で生成した合成 Ordinal データ．
synthetic_data_release の評価フレームワークにそのまま入力できる形式
（`*.csv` + `*.json` メタデータ）で配置している．

## 生成方法

1. ランク `r` の潜在信号行列 `U @ V^T` を作成 (`U ~ N(0, 1/r)` の `n x r`，`V ~ N(0, 1/r)` の `p x r`)．
2. SNR=2.0 となるガウスノイズを加える．
3. 各列について分位点で `K=5` ビンに離散化し，値域 `{0,1,2,3,4}` の整数にする．

これで理論ランクが `r`，stable rank もおよそ `r` に近い低ランク構造を持つ離散データが得られる．

## 提供するデータセット

すべて `n=20000` 行，`p=10` 列，`K=5` ビン，列名 `attr_0 ... attr_9`．

| 名前              | ランク `r` | seed | 位置付け                     |
|-------------------|-----------|------|------------------------------|
| `eff_rank_best`   | 3         | 42   | 低 effective rank（Mondrian 有利） |
| `eff_rank_worst`  | 10        | 43   | フルランク（Mondrian 不利）       |

movielens の `eff_rank_best` / `eff_rank_worst` と同じ「Mondrian の得意・不得意を対比する」
位置付けを，純粋な合成データで再現したもの．

## ファイル構成

```
data/synthetic_lowrank/
├── README.md
├── generate_synthetic_lowrank.py   # 再現用スクリプト（seed 固定）
├── eff_rank_best.csv               # 20000 行 x 10 列，Ordinal (0..4)
├── eff_rank_best.json              # メタデータ
├── eff_rank_worst.csv              # 20000 行 x 10 列，Ordinal (0..4)
└── eff_rank_worst.json             # メタデータ
```

## 再生成

```
python generate_synthetic_lowrank.py
```

seed は固定済みなので，何度実行しても同一のデータが得られる．
行数・列数などを変えたい場合は引数で指定できる．

```
python generate_synthetic_lowrank.py --n 50000 --p 10 --K 5 --snr 2.0
```

## runconfig

対応する runconfig は以下に配置済み:

```
tests/linkage/runconfig.synthetic_lowrank_eff_rank_best.json
tests/linkage/runconfig.synthetic_lowrank_eff_rank_worst.json
tests/inference/runconfig.synthetic_lowrank_eff_rank_best.json
tests/inference/runconfig.synthetic_lowrank_eff_rank_worst.json
tests/utility/runconfig.synthetic_lowrank_eff_rank_best.json
tests/utility/runconfig.synthetic_lowrank_eff_rank_worst.json
```

実験実行例:

```
python linkage_cli.py    --datapath data/synthetic_lowrank/eff_rank_best  \
                         --runconfig tests/linkage/runconfig.synthetic_lowrank_eff_rank_best.json
python inference_cli.py  --datapath data/synthetic_lowrank/eff_rank_best  \
                         --runconfig tests/inference/runconfig.synthetic_lowrank_eff_rank_best.json
python utility_cli.py    --datapath data/synthetic_lowrank/eff_rank_best  \
                         --runconfig tests/utility/runconfig.synthetic_lowrank_eff_rank_best.json
```
