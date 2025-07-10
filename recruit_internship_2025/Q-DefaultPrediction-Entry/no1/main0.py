import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

INPUT_DIR = "../data"


def write_submission(id_list, pred, output_file):
    """予測結果を出力します."""
    df = pd.DataFrame({'id': id_list, 'default': pred})
    df.to_csv(output_file, index=False)


def preprocess(df, mms=None, fit_scaler=False):
    # emp_length の変換
    transform_emp = {'< 1 year':0.0,'1 year':1.0,'2 years':2.0,'3 years':3.0,'4 years':4.0,
                     '5 years':5.0,'6 years':6.0,'7 years':7.0,'8 years':8.0,'9 years':9.0,
                     '10 years':10.0,'10+ years':11.0}
    df['emp_length'] = df['emp_length'].fillna('1 year').map(transform_emp)

    # term の変換
    transform_term = {'36 months':36.0,'60 months':60.0,}
    df['term'] = df['term'].fillna('36 months').map(transform_term)

    # カテゴリカル変数の変換
    categorical_cols = ['emp_title', 'home_ownership', 'verification_status', 'purpose']
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]

    # 特徴量の追加
    df['income_to_loan_ratio'] = df['annual_inc'] / df['loan_amnt']
    df['emp_length_to_income'] = df['emp_length'] * df['annual_inc']
    df['int_rate_installment'] = df['int_rate'] * df['installment']

    # 数値カラム抽出、欠損をカラム平均で補完
    num_mask = [is_numeric_dtype(dt) for dt in df.dtypes]
    feats = df.loc[:, num_mask].fillna(df.loc[:, num_mask].mean())

    # スケーリング
    if fit_scaler:
        mms = MinMaxScaler().fit(feats)
    feats_scaled = mms.transform(feats)
    return feats_scaled, mms


def adversarial_validation(train_X, test_X):
    X = np.vstack([train_X, test_X])
    y_domain = np.array([0] * len(train_X) + [1] * len(test_X))

    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(X, y_domain)
    # テストドメインと判定される確率を返す (1に近いほどtestに似る)
    prob = rf.predict_proba(train_X)[:, 1]
    return prob


def main():
    # データ読み込み
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train/train.csv'))
    y = train_df['default'].values
    train_df = train_df.drop('default', axis=1)
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test/test.csv'))
    test_ids = test_df['id'].values

    # 前処理
    X_train_all, mms = preprocess(train_df, fit_scaler=True)
    X_test, _ = preprocess(test_df, mms=mms, fit_scaler=False)

    # Adversarial Validation
    domain_scores = adversarial_validation(X_train_all, X_test)

    # テストに近いサンプルを重点的に学習
    sample_weights = domain_scores  + 0.2

    # クロスバリデーション設定 (本番環境ではアンサンブル)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    params = {
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [100, 125, 150, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'lambda': [0.5, 1.0]
    }
    scoring = 'roc_auc'

    test_pred_sum = np.zeros(len(X_test))

    # 各foldで学習&推論
    for _fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_all, y), 1):
        X_tr, _X_val = X_train_all[tr_idx], X_train_all[val_idx]
        y_tr, _y_val = y[tr_idx], y[val_idx]
        w_tr = sample_weights[tr_idx]

        # GridSearchCV に 標本重みを渡す
        grid = GridSearchCV(
            estimator=XGBClassifier(random_state=42, objective = 'binary:logistic',
                eval_metric='logloss'),
            param_grid=params,
            scoring=scoring,
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_tr, y_tr, sample_weight=w_tr)

        test_pred_sum += grid.predict_proba(X_test)[:, 1]


    # 最終予測
    final_test_pred = test_pred_sum / kf.n_splits

    gt_df = pd.read_csv(os.path.join(INPUT_DIR, 'groundtruth/ground_truth.csv'))
    auc_score = roc_auc_score(gt_df['default'], final_test_pred)
    print(f"テストデータでのAUCスコア: {auc_score:.4f}")

    write_submission(test_ids, final_test_pred, os.path.join(INPUT_DIR, 'submission.csv'))


if __name__ == '__main__':
    main()
