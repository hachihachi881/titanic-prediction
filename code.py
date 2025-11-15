import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv(r"c:\Users\EGC\kaggle\titanic\train.csv")
test = pd.read_csv(r"c:\Users\EGC\kaggle\titanic\test.csv")

# print(train.head())

test_shape = test.shape
train_shape = train.shape

# --- 前処理: 欠損値補完とカテゴリ変換 + 特徴量エンジニアリング ---

# データを結合して一括処理
all_data = [train, test]

for dataset in all_data:
    # Age の欠損値を中央値で補完
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    # Fare の欠損値を中央値で補完
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    # Embarked の欠損値を最頻値で補完
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    
    # Sex を数値に変換
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # Embarked を数値に変換
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    # 家族の人数を計算 (SibSp: 兄弟姉妹/配偶者, Parch: 親/子供)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # 一人かどうか
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    
    # 名前から敬称（Title）を抽出
    dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # 敬称をグループ化
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                   'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                   'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Title を数値に変換
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'].fillna(0, inplace=True)
    dataset['Title'] = dataset['Title'].astype(int)
    
    # 年齢を区分化
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5, labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 運賃を区分化
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)

def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns

# 使用する特徴量を選択
feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features = train[feature_cols].values

# RandomForest モデルの作成（決定木より精度が高い）
model = RandomForestClassifier(
    n_estimators=100,      # 決定木の数
    max_depth=5,           # 木の深さ（過学習を防ぐ）
    min_samples_split=4,   # 分割に必要な最小サンプル数
    min_samples_leaf=2,    # 葉ノードの最小サンプル数
    random_state=42        # 再現性のため
)
model.fit(features, target)

# クロスバリデーションで精度を確認
cv_scores = cross_val_score(model, features, target, cv=5)
print(f"\n=== モデルの精度評価 ===")
print(f"クロスバリデーション精度: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"各fold: {cv_scores}")

# 「test」の説明変数を使って予測
test_features = test[feature_cols].values
my_prediction = model.predict(test_features)

print(f"\n=== 予測結果 ===")
print(f"生存予測数: {my_prediction.sum()} / {len(my_prediction)}")
print(f"生存率: {my_prediction.mean():.2%}")

# Kaggle提出用のCSVファイルを作成
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": my_prediction
})
submission.to_csv(r"c:\Users\EGC\kaggle\titanic\submission.csv", index=False)
print(f"\n提出ファイルを保存しました: submission.csv")
print(f"行数: {len(submission)}")
print(submission.head(10))


# train["Age"] = train["Age"].fillna(train["Age"].median())
# train["Embarked"] = train["Embarked"].fillna("S")

# train["Sex"][train["Sex"] == "male"] = 0
# train["Sex"][train["Sex"] == "female"] = 1
# train["Embarked"][train["Embarked"] == "S" ] = 0
# train["Embarked"][train["Embarked"] == "C" ] = 1
# train["Embarked"][train["Embarked"] == "Q"] = 2

# test["Age"] = test["Age"].fillna(test["Age"].median())
# test["Embarked"] = test["Embarked"].fillna("S")

# test["Sex"][test["Sex"] == "male"] = 0
# test["Sex"][test["Sex"] == "female"] = 1
# test["Embarked"][test["Embarked"] == "S" ] = 0
# test["Embarked"][test["Embarked"] == "C" ] = 1
# test["Embarked"][test["Embarked"] == "Q"] = 2

# test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# train.head(10)

# print(train.head())
# print(train.head(20))

# print(test_shape)
# print(train_shape)

# print(test.describe())
# print(train.describe())

# print(kesson_table(train))
# print(kesson_table(test))