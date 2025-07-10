import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


INPUT_DIR = "../data"


def write_submission(id_list, pred, output_file):
    """予測結果を出力します."""
    df = pd.DataFrame({'id': id_list, 'default': pred})
    df.to_csv(output_file, index=False)


def adversarial_validation(train_df, test_df):
    """訓練データとテストデータの分布の違いを検出します"""
    # 訓練データとテストデータを結合
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # 目的変数と説明変数を分離
    y = combined_df['is_train']
    X = combined_df.drop('is_train', axis=1)
    
    # データを分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデルの学習
    clf = XGBClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # 予測確率を計算
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    
    print(f'Adversarial Validation AUC: {auc_score:.4f}')
    
    # 特徴量の重要度を表示
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\n特徴量の重要度（上位10件）:")
    print(feature_importance.head(10))
    
    return auc_score


def main():
    train_file = os.path.join(INPUT_DIR, 'train/train.csv')
    target_name = 'default'

    # 目的変数、説明変数を抽出します.
    train_df = pd.read_csv(train_file)
    targets = train_df[target_name]
    train_df = train_df.drop(target_name, axis=1)

    # print(train_df)
    
    # 特徴量エンジニアリング
    transform_emp = {'< 1 year':0.0,'1 year':1.0,'2 years':2.0,'3 years':3.0,'4 years':4.0,
    '5 years':5.0,'6 years':6.0,'7 years':7.0,'8 years':8.0,'9 years':9.0,'10 years':10.0,
    '10+ years':11.0}

    # 欠損値を処理してから変換
    train_df['emp_length'] = train_df['emp_length'].fillna(train_df['emp_length'].mode())  # 欠損値を'< 1 year'で補完
    train_df['emp_length'] = list(map(lambda x: transform_emp[x], train_df['emp_length']))

    # カテゴリカル変数の変換
    categorical_cols = ['emp_title', 'home_ownership', 'verification_status', 'purpose']
    for col in categorical_cols:
        train_df[col] = pd.factorize(train_df[col])[0]

    # print(train_df)
    
    # 2. 新しい特徴量の作成
    train_df['income_to_loan_ratio'] = train_df['annual_inc'] / train_df['installment']
    train_df['income_to_loan_ratio'] = train_df['annual_inc'] / train_df['loan_amnt']
    train_df['emp_length_to_income'] = train_df['emp_length'] * train_df['annual_inc']
    train_df['int_rate_installment'] = train_df['int_rate'] * train_df['installment']  # 新しい特徴量を追加
    
    num_cols = [is_numeric_dtype(dtype) for dtype in train_df.dtypes]
    train_features = train_df.loc[:, num_cols]
    train_features = train_features.fillna(train_features.mean())
    mms = MinMaxScaler()
    train_features = mms.fit_transform(train_features)

    # モデルの学習を行います.
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, targets, test_size=0.2, random_state=42
        )
    clf = XGBClassifier(random_state=42)
    scoring = 'roc_auc'
    params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 125, 150, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    model = GridSearchCV(
            clf,
            params,
            scoring=scoring,
            cv=5,
            n_jobs=-1
            )
    model.fit(X=X_train, y=y_train)
    
    # 検証データでのAUCスコアを計算
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    print(f'検証データでのAUCスコア: {auc_score:.4f}')
    print(f'最適なパラメータ: {model.best_params_}')

    # テストデータの読み込みと前処理
    test_file = os.path.join(INPUT_DIR, 'test/test.csv')
    test_df = pd.read_csv(test_file)
    
    # 訓練データとテストデータの前処理
    for df in [train_df, test_df]:
        df['emp_length'] = df['emp_length'].fillna('< 1 year')
        df['emp_length'] = list(map(lambda x: transform_emp[x], df['emp_length']))
        
        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]
        
        df['income_to_loan_ratio'] = df['annual_inc'] / df['installment']
        df['income_to_loan_ratio'] = df['annual_inc'] / df['loan_amnt']
        df['emp_length_to_income'] = df['emp_length'] * df['annual_inc']
        df['int_rate_installment'] = df['int_rate'] * df['installment']
    
    # Adversarial Validationの実行
    print("\nAdversarial Validationの実行:")
    adv_auc = adversarial_validation(train_df, test_df)
    
    if adv_auc > 0.7:
        print("\n警告: 訓練データとテストデータの分布に大きな違いがあります")
    else:
        print("\n訓練データとテストデータの分布は比較的似ています")

    # 予測を行います.
    test_df = pd.read_csv(test_file)
    pred = model.predict_proba(test_df.loc[:, num_cols])[:, 1]

    # 結果を出力します.
    output_file = os.path.join(INPUT_DIR, 'submission.csv')
    write_submission(test_df['id'], pred, output_file)



if __name__ == '__main__':
    main()


