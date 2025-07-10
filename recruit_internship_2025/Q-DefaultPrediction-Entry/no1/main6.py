import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# データの読み込み
train_df = pd.read_csv('../data/train/train.csv')
test_df = pd.read_csv('../data/test/test.csv')

# 目的変数の分離
targets = train_df['default']
train_df = train_df.drop('default', axis=1)

# 前処理関数の定義（例：欠損値処理、カテゴリ変数のエンコーディングなど）
def preprocess(df):
    # 例として、数値型の列のみを抽出し、欠損値を平均で補完
    num_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    df = df[num_cols]
    df = df.fillna(df.mean())
    return df

# 前処理の適用
train_processed = preprocess(train_df)
test_processed = preprocess(test_df)

# スケーリング
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_processed)
test_scaled = scaler.transform(test_processed)



# トレーニングデータとテストデータの結合
X_adv = np.vstack((train_scaled, test_scaled))
y_adv = np.hstack((np.zeros(train_scaled.shape[0]), np.ones(test_scaled.shape[0])))

# 分類モデルの学習
adv_clf = RandomForestClassifier(n_estimators=100, random_state=42)
adv_clf.fit(X_adv, y_adv)

# トレーニングデータに対するテストデータらしさの予測確率
train_proba = adv_clf.predict_proba(train_scaled)[:, 1]

# テストデータに類似したトレーニングデータの選択（上位30%を選択）
threshold = np.percentile(train_proba, 70)
similar_idx = np.where(train_proba >= threshold)[0]
X_similar = train_scaled[similar_idx]
y_similar = targets.iloc[similar_idx]




# モデルの定義
model = XGBClassifier(random_state=42)

# クロスバリデーションによる評価
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(model, X_similar, y_similar, cv=cv, scoring='roc_auc')

print(f'平均AUCスコア: {np.mean(auc_scores):.4f}')


