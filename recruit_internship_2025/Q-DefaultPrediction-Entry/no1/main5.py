import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

INPUT_DIR = "../data"

def write_submission(id_list, pred, output_file):
    """予測結果を出力します."""
    df = pd.DataFrame({'id': id_list, 'default': pred})
    df.to_csv(output_file, index=False)

def preprocess(df, mms=None, fit_scaler=False):
    """train/test共通の前処理を行い、NumPy 配列を返す。"""
    # emp_length のマッピング
    transform_emp = {'< 1 year':0.0,'1 year':1.0,'2 years':2.0,'3 years':3.0,'4 years':4.0,
                     '5 years':5.0,'6 years':6.0,'7 years':7.0,'8 years':8.0,'9 years':9.0,
                     '10 years':10.0,'10+ years':11.0}
    df['emp_length'] = df['emp_length'].fillna('1 year')
    df['emp_length'] = df['emp_length'].map(transform_emp)

    # カテゴリカル変数をファクター化
    categorical_cols = ['emp_title', 'home_ownership', 'verification_status', 'purpose']
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]

    # 新特徴量
    df['income_to_loan_ratio'] = df['annual_inc'] / df['loan_amnt']
    df['emp_length_to_income'] = df['emp_length'] * df['annual_inc']
    df['int_rate_installment'] = df['int_rate'] * df['installment']

    # 数値カラムのみ抽出し、欠損を平均値で補完
    num_mask = [is_numeric_dtype(dt) for dt in df.dtypes]
    feats = df.loc[:, num_mask].fillna(df.loc[:, num_mask].mean())

    # スケーリング
    if fit_scaler:
        mms = MinMaxScaler().fit(feats)
    feats_scaled = mms.transform(feats)
    return feats_scaled, mms

def main():
    # ----- データ読み込み -----
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train/train.csv'))
    targets   = train_df['default'].values
    train_df  = train_df.drop('default', axis=1)
    test_df   = pd.read_csv(os.path.join(INPUT_DIR, 'test/test.csv'))
    test_ids  = test_df['id'].values

    # ----- 前処理 -----
    X_all, mms = preprocess(train_df, fit_scaler=True)
    X_test, _  = preprocess(test_df,  mms=mms, fit_scaler=False)

    # ----- クロスバリデーション設定 -----
    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    params = {
        'max_depth': [2, 3],
        'learning_rate': [0.05],
        'n_estimators': [100, 150],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    scoring = 'roc_auc'

    # フォールド毎にモデルを学習／予測
    test_pred_sum = np.zeros(len(X_test))
    aucs = []
    # preds = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all, targets), 1):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = targets[tr_idx], targets[val_idx]

        # グリッドサーチ
        grid = GridSearchCV(
            estimator=XGBClassifier(random_state=42),
            param_grid=params,
            scoring=scoring,
            cv=3,          # 内部でさらに 3-fold CV
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_tr, y_tr)

        # 検証セットでの評価
        val_pred = grid.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        print(f'Fold{fold} AUC: {auc:.4f}  (best_params: {grid.best_params_})')
        aucs.append(auc)
        # pred = grid.predict_proba(test_df)[:, 1]
        # preds.append(pred)

        # テストセット予測を累積
        test_pred_sum += grid.predict_proba(X_test)[:, 1]

    # ----- 結果の集計 -----
    mean_auc = np.mean(aucs)
    print(f'\n5-Fold CV 平均 AUC: {mean_auc:.4f}')

    # pred_mean = np.mean(preds)
    gt_file = os.path.join(INPUT_DIR, 'groundtruth/ground_truth.csv')
    gt_df = pd.read_csv(gt_file)
    gt_df = gt_df['default']
    # auc_score = roc_auc_score(gt_df, pred_mean)
    # print(f'テストデータでのAUCスコア: {auc_score:.4f}')

    # テスト予測は 5 モデルの平均
    final_test_pred = test_pred_sum / kf.n_splits

    auc_score = roc_auc_score(gt_df, final_test_pred)
    print(f'テストデータでのAUCスコア: {auc_score:.4f}')

    # 提出用ファイル出力
    output_file = os.path.join(INPUT_DIR, 'submission.csv')
    write_submission(test_ids, final_test_pred, output_file)
    print(f'提出ファイルを出力しました: {output_file}')

if __name__ == '__main__':
    main()
