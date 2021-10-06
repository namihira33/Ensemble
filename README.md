# 眼科AIコンテスト
 
眼科AIコンテストの年齢推定コンペに向けたプログラムを載せるレポジトリです。

# 目標

年齢推定で誤差を減らすこと
MAE2.5を切ることが目標

# 優先順位
1. MAEを減少させる。
2. 医学的に意義のある解析方法を採用する。
 
# Requirement
 
Poetry.tomlに記述されています。
* python = "^3.8"
* matplotlib = "^3.4.2"
* torch = "^1.9.0"
* torchvision = "^0.10.0"
* sklearn = "^0.0"
* tqdm = "^4.61.2"
* ipykernel = "^6.0.1"
* skorch = "^0.10.0"
* pandas = "^1.3.3"
* seaborn = "^0.11.2"
 
# Installation
 
```bash
poetry update
```
 
# Usage
 
```bash
cd ~/opth_aicontest
python ./src/run.py (モデル名)
python ./src/evaluate.py (数字)
```

* モデル名はResnet34などが入ります。
* 数字を指定することで、n回目のモデルを評価できます。
 
# 取り組む課題
 
* JOIレジストリから、全身的に健康な人のデータを抜き出す。
* 眼科AI学会配布のデータ「眼科AIデータ」とJOIからのデータ「JOIデータ」は区別する。
* 学習率・epoch数で、回帰モデルのハイパラをチューニングする。
* 上を使って、DataAugmentationの有効性を検討する。
* ドメイン知識を実装して、上で有効性を検討する。
* 分類の活かし方ßを決める。
* アンサンブル学習のコード実装
* 何と何を組み合わせて使うのかを決める。