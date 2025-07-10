import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

INPUT_DIR = "../data"


def write_submission(id_list, pred, output_file):
    """予測結果を出力します."""
    df = pd.DataFrame({'id': id_list, 'default': pred})
    df.to_csv(output_file, index=False)


def preprocess(df, mms=None, fit_scaler=False):
    """共通の前処理: emp_length 変換、カテゴリfactor化、新特徴量、スケーリング"""
    # emp_length のマッピング
    transform_emp = {'< 1 year':0.0,'1 year':1.0,'2 years':2.0,'3 years':3.0,'4 years':4.0,
                     '5 years':5.0,'6 years':6.0,'7 years':7.0,'8 years':8.0,'9 years':9.0,
                     '10 years':10.0,'10+ years':11.0}
    df['emp_length'] = df['emp_length'].fillna('1 year').map(transform_emp)

    transform_term = {'36 months':36.0,'60 months':60.0,}
    df['term'] = df['term'].fillna('36 months').map(transform_term)

    # カテゴリカル変数を factorize
    categorical_cols = ['emp_title', 'home_ownership', 'verification_status', 'purpose']
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]

    # 新特徴量
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


# def adversarial_validation(train_X, test_X):
#     X = np.vstack([train_X, test_X])
#     y_domain = np.array([0] * len(train_X) + [1] * len(test_X))

#     rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
#     rf.fit(X, y_domain)
#     # テストドメインと判定される確率を返す (1 に近いほど test に似る)
#     prob = rf.predict_proba(train_X)[:, 1]
#     return prob

def adversarial_validation(train_X, test_X):
    n = 20.0
    # testデータとの類似率上位n%を出力
    X = np.vstack([train_X, test_X])
    y_domain = np.array([0] * len(train_X) + [1] * len(test_X))

    # Train a classifier to distinguish between train and test sets
    # rf = RandomForestClassifier(n_estimators=100, random_state=523, n_jobs=-1)
    rf = XGBClassifier(random_state=42, objective = 'binary:logistic',
            eval_metric='logloss')
    rf.fit(X, y_domain)

    # Predict probabilities of being from the test set
    prob = rf.predict_proba(train_X)[:, 1]

    # Get the threshold for the top n% most test-like instances
    threshold = np.percentile(prob, 100.0 - n)
    # threshold = 0.50
    # threshold = np.array([0.50] * len(prob))

    # Return the indices and probabilities of top 20% most test-like samples in train_X
    top_mask = prob >= threshold
    top_indices = np.where(top_mask)[0]
    top_probs = prob[top_mask]
    
    other_mask = prob < threshold
    other_indices = np.where(other_mask)[0]
    other_probs = prob[other_mask]
    

    return top_indices, top_probs, other_indices, other_probs


def main():
    # データ読み込み
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train/train.csv'))
    train_y = train_df['default'].values
    train_df = train_df.drop('default', axis=1)
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test/test.csv'))
    test_ids = test_df['id'].values

    # 前処理
    train_x, mms = preprocess(train_df, fit_scaler=True)
    test_x, _ = preprocess(test_df, mms=mms, fit_scaler=True)

    # Adversarial Validation
    vals_indices, w_vals, train_indices, w_trains = adversarial_validation(train_x, test_x)
    w_trains = w_trains * 10.0 + 0.001
    vals_x = train_x[vals_indices]
    vals_y = train_y[vals_indices]
    train_x = train_x[train_indices]
    train_y = train_y[train_indices]

    params = {
        'max_depth': [2, 3],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [100, 125, 150, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        # 'alpha': [0.01,0.05,0.1]
        'lambda': [0.5, 1.0]
    }
    # scoring = 'roc_auc'

    # test_pred_sum = np.zeros(len(X_test))
    # aucs = []

    # GridSearchCV に 標本重みを渡す
    grid = GridSearchCV(
        estimator=XGBClassifier(random_state=42, objective = 'binary:logistic',
            eval_metric='logloss'),
        param_grid=params,
        scoring='roc_auc',
        # cv=ps,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(vals_x, vals_y, sample_weight=w_trains)

    print(f'best_params: {grid.best_params_}')

    # val_pred = grid.predict_proba(X_val)[:, 1]
    # auc = roc_auc_score(y_val, val_pred)
    # print(f'Fold{fold} AUC: {auc:.4f} (best_params: {grid.best_params_})')
    # aucs.append(auc)
    # test_pred_sum += grid.predict_proba(X_test)[:, 1]

    # mean_auc = np.mean(aucs)
    # print(f"\n5-Fold CV 平均 AUC: {mean_auc:.4f}")

    # 最終予測
    # final_test_pred = test_pred_sum / kf.n_splits
    
    pred = grid.predict_proba(test_x)[:, 1]
    
    # 真値で評価
    gt_df = pd.read_csv(os.path.join(INPUT_DIR, 'groundtruth/ground_truth.csv'))
    auc_score = roc_auc_score(gt_df['default'], pred)
    print(f"テストデータでのAUCスコア: {auc_score:.4f}")

    # 提出ファイル出力
    write_submission(test_ids, pred, os.path.join(INPUT_DIR, 'submission.csv'))
    print("提出ファイルを出力しました: submission.csv")


if __name__ == '__main__':
    main()
