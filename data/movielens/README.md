# MovieLens データセット

MovieLens ml-32m から生成したユーザー×ジャンル集約テーブル．1 行 = 1 ユーザー，列 = ジャンル，値 = そのユーザーがそのジャンルに対して付けた平均レーティング（0-10 の整数 Ordinal，未評価は 0）．

前処理は別リポジトリ [SDC4Society/MovieLensPreprocessing](https://github.com/SDC4Society/MovieLensPreprocessing) で行っている．本ディレクトリはその成果物を synthetic data release の評価用に配置したもの．

---

## ディレクトリ構成

```
data/movielens/
├── README.md
├── ml32m_full.csv                       # 行=200,948, 列=20 (全ジャンル)
├── ml32m_full.json                      # メタデータ
├── ml32m_full_rareUserIDs.csv           # 各ユーザーのレア属性カウント（降順）
└── ncols_10/
    ├── eff_rank_best.csv                # 行=200,948, 列=10（上位10ジャンル）
    ├── eff_rank_best.json
    ├── eff_rank_best_rareUserIDs.csv
    ├── eff_rank_worst.csv               # 行=200,948, 列=10（下位10ジャンル）
    ├── eff_rank_worst.json
    └── eff_rank_worst_rareUserIDs.csv
```

提供データセットは 3 種類: `ml32m_full`, `ncols_10/eff_rank_best`, `ncols_10/eff_rank_worst`．

`ncols_20`（= 全ジャンル）では best / worst 選抜が恒等となり両者が一致するため，単一の `ml32m_full` に統合している．

---

## ファイルの意味

### `*.csv`

- 行 = ユーザー（ratings.csv の userId を昇順 unique 配列化した順番．行 `i` ⇔ `uids[i]`）
- 列 = ジャンル
  - `ml32m_full`: MovieLens ml-32m の全 20 ジャンル
  - `ncols_10/eff_rank_best`: ユーザー×ジャンル評価行列の**有効ランクへの寄与が大きい**上位 10 ジャンル
  - `ncols_10/eff_rank_worst`: 同じく**寄与が小さい**下位 10 ジャンル
- 値 = そのユーザーがそのジャンルに付けた平均レーティング（rating × 2 を四捨五入した 0-10 の整数．未評価 NaN は 0 に置換）

### `*.json`

データローダ用のメタデータ．全列 `Ordinal`，`size=11`，`i2s=["0",...,"10"]`．`utils/datagen.load_local_data_as_df()` が CSV と対で読み込む．

### `*_rareUserIDs.csv`

各ユーザーが持つ「レア値」の本数をカウントし，多い順（次いで userId 昇順）にソートしたファイル．

- 列: `userId`, `rarity_count`
- レア値定義: 各列で出現割合 ≤ 5% の値（デフォルト閾値）．**0（＝未評価）も頻度 ≤ 5% ならレア値**扱い．
- `rarity_count` 上限 = 列数（`ml32m_full` なら 20，`ncols_10/*` なら 10）．全列でレア値を持つユーザーを「最もレアなユーザー」とみなす．

`userId` はカウント列なので，合成データ評価の `Targets` 指定（`ID<row_index>` 形式）に渡す際は行番号への変換が必要．変換式は:

```python
sorted_uids = np.sort(rareUserIDs['userId'].values)
row_index   = int(np.searchsorted(sorted_uids, target_userId))
Target_ID   = f"ID{row_index}"
```

`rareUserIDs.csv` には全ユーザーが含まれるため，`userId` 列を昇順ソートすれば元の `uids` 配列が復元できる．

---

## 統計サマリ

| データセット | 行数 | 列数 | 最大 rarity_count | 最大値該当ユーザー数 |
|---|---|---|---|---|
| `ml32m_full`              | 200,948 | 20 | 20 | 40 |
| `ncols_10/eff_rank_best`  | 200,948 | 10 | 10 | 50 |
| `ncols_10/eff_rank_worst` | 200,948 | 10 | 10 | 988 |

---

## 生成手順

CSV / JSON / rareUserIDs はすべて前処理リポジトリ [SDC4Society/MovieLensPreprocessing](https://github.com/SDC4Society/MovieLensPreprocessing) 側で生成している．再生成する場合は前処理リポジトリの README を参照．

---

## CLI での指定

`-D` オプションには拡張子なしのパスを渡す:

```bash
uv run python linkage_cli.py \
    -D data/movielens/ml32m_full \
    -RC tests/linkage/movielens/runconfig.ml32m_full.default.json \
    -O tests/linkage

uv run python linkage_cli.py \
    -D data/movielens/ncols_10/eff_rank_best \
    -RC tests/linkage/movielens/ncols_10/runconfig.eff_rank_best.default.json \
    -O tests/linkage
```
