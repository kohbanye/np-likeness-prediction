# CLAUDE.md

このプロジェクトでは、天然物らしさを予測するモデルを構築します。

## データセット

データセットは天然物と合成化合物の SMILES 文字列のリストです。天然物に似ていて、合成化合物に似ていないものを天然物らしいとして予測することを目指します。

天然物データセットは以下のようにして、COCONUT から取得できます。zip を解凍すると、CSV ファイルが得られます。CSV のカラムに"canonical_smiles"があるので、これを抽出して、データセットとすると良いでしょう。

```shell
curl 'https://coconut.s3.uni-jena.de/prod/downloads/2025-08/coconut_csv-08-2025.zip'
```

合成化合物データセットは以下のようにして、count, subset (none で良い) を指定して ZINC22 から取得できます。これにより得た txt ファイルには、SMILES や化合物の ID が含まれているので、SMILES のみ抽出してデータセットとします。

```shell
curl 'https://cartblanche.docking.org/substance/random.txt' \
  -H 'Accept: application/json, text/plain, */*' \
  -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundary0cFylzxojzDWwpB0' \
  --data-raw $'------WebKitFormBoundary0cFylzxojzDWwpB0\r\nContent-Disposition: form-data; name="count"\r\n\r\n1000000\r\n------WebKitFormBoundary0cFylzxojzDWwpB0\r\nContent-Disposition: form-data; name="subset"\r\n\r\nnone\r\n------WebKitFormBoundary0cFylzxojzDWwpB0--\r\n'
```

データセットは `data/` ディレクトリに保存するようにしてください。

## モデル

モデルには、GPT 系の decoder-only Transformer を使用します。あるデータセットで学習した言語モデルに SMILES を入力した際の perplexity の値はその化合物が学習データセットに含まれそうかどうかの指標になります。このようにして、天然物データセットで学習したモデルと合成化合物データセットで学習したモデルを比較することで、天然物らしさを予測します。

## スコア

天然物らしさスコアは、対数尤度比を用いて計算します。具体的には、以下の式で計算されます。

$$
\text{score} = = \log \frac{P(\text{natural} | x)}{P(\text{synth} | x)} \\
= \log \frac{P(x | \text{natural})}{P(x | \text{synth})} + \log \frac{P(\text{natural})}{P(\text{synth})} \\
= \log P(x | \text{natural}) - \log P(x | \text{synth}) + \log \frac{P(\text{natural})}{P(\text{synth})}
$$

$\frac{P(\text{natural})}{P(\text{synth})}$ は事前確率であり、データセットのバランスに依存します。ここでは、天然物と合成化合物のデータセットが同じサイズであると仮定し、事前確率は 1 とします。

## 実装

transformers, pytorch-lightning を使用して実装します。モデルは Llama 3 と GPT-2 を使い、比較を行います。
ライブラリのインストールには uv を使用します。lint, format は ruff を使用し、型チェックには ty を使用します。
ソースコードは `src/` ディレクトリに配置し、学習用のスクリプトなどは `scripts/` ディレクトリに配置します。
