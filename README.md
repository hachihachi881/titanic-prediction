# Kaggle Titanic Survival Prediction

Kaggle Titanicコンペティションの生存予測モデル

## 概要
タイタニック号の乗客データを使用して、生存率を予測する機械学習モデルです。

## モデル
- **アルゴリズム**: Random Forest Classifier
- **精度**: 82.83% (クロスバリデーション)
- **特徴量**: 
  - 基本情報: Pclass, Sex, Age, Fare, Embarked
  - エンジニアリング特徴量: FamilySize, IsAlone, Title, AgeBin, FareBin

## ファイル構成
- `code.py`: メインの予測スクリプト
- `train.csv`: 訓練データ
- `test.csv`: テストデータ
- `submission.csv`: 提出用の予測結果

## 使い方
```bash
# 必要なパッケージをインストール
pip install pandas numpy scikit-learn

# 予測を実行
python code.py
```

## 結果
- 生存予測数: 156 / 418
- 予測生存率: 37.32%
