import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# データ読み込み
def load_data():
    try:
        train = pd.read_csv("../data/train/train.csv")
        test = pd.read_csv("../data/test/test.csv")
        print(f"データ読み込み完了: 訓練データ {train.shape}, テストデータ {test.shape}")
        return train, test
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}")
        raise

# カテゴリカル変数の前処理
def preprocess(df):
    df = df.copy()
    print("前処理開始...")
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna("missing", inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"カテゴリカル変数 '{col}' の処理完了")
    
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"数値変数 '{col}' の処理完了")
    return df

# Adversarial Validation
def adversarial_validation(train_df, test_df):
    print("Adversarial Validation開始...")
    train_adv = train_df.copy()
    test_adv = test_df.copy()
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    test_adv['default'] = 0  # ダミー

    df = pd.concat([train_adv, test_adv], sort=False)
    df = preprocess(df.drop(columns=['id']))

    X = df.drop(columns=['default', 'is_test'])
    y = df['is_test']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Adversarial Validation AUC: {auc:.4f}")
    
    importance = model.feature_importances_
    return [col for col, imp in zip(X.columns, importance) if imp > 0]

# モデル学習と予測
def train_and_predict(train_df, test_df, features):
    print(f"選択された特徴量: {len(features)}個")
    X_train = preprocess(train_df[features])
    y_train = train_df['default']
    X_test = preprocess(test_df[features])

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("モデル学習開始...")
    model.fit(X_train, y_train)
    print("予測開始...")
    preds = model.predict_proba(X_test)[:, 1]
    return preds

# submission作成
def save_submission(test_df, preds):
    try:
        submission = pd.DataFrame({"id": test_df['id'], "default": preds})
        os.makedirs("../data", exist_ok=True)
        submission.to_csv("../data/submission.csv", index=False)
        print("submissionファイルの保存が完了しました")
    except Exception as e:
        print(f"エラー: submissionファイルの保存に失敗しました - {e}")
        raise

if __name__ == "__main__":
    try:
        print("処理開始...")
        train_df, test_df = load_data()
        selected_features = adversarial_validation(train_df, test_df)
        predictions = train_and_predict(train_df, test_df, selected_features)
        save_submission(test_df, predictions)
        print("全ての処理が完了しました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise

