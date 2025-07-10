import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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


def adversarial_validation(train_X, test_X):
    n = 100.0
    # testデータとの類似率上位n%を出力
    X = np.vstack([train_X, test_X])
    y_domain = np.array([0] * len(train_X) + [1] * len(test_X))

    # Train a classifier to distinguish between train and test sets
    rf = RandomForestClassifier(n_estimators=100, random_state=523, n_jobs=-1)
    # rf = XGBClassifier(random_state=42, objective = 'binary:logistic',
    #         eval_metric='logloss')
    rf.fit(X, y_domain)

    # Predict probabilities of being from the test set
    prob = rf.predict_proba(train_X)[:, 1]

    # Get the threshold for the top n% most test-like instances
    threshold = np.percentile(prob, 100.0 - n)
    # print(threshold)
    # threshold = 0.50
    # threshold = np.array([0.50] * len(prob))

    # Return the indices and probabilities of top 20% most test-like samples in train_X
    top_mask = prob >= threshold
    top_indices = np.where(top_mask)[0]
    top_probs = prob[top_mask]
    
    # print(top_probs)

    return top_indices, top_probs


def main():
    train_file = os.path.join(INPUT_DIR, 'train/train.csv')
    target_name = 'default'

    # 目的変数、説明変数を抽出します.
    train_df = pd.read_csv(train_file)
    targets = train_df[target_name]
    train_df = train_df.drop(target_name, axis=1)
    num_cols = [is_numeric_dtype(dtype) for dtype in train_df.dtypes]
    train_features = train_df.loc[:, num_cols]
    train_features = train_features.fillna(train_features.mean())
    mms = MinMaxScaler()
    train_features = mms.fit_transform(train_features)
    preprocess(train_features)
    
    # 予測を行います.
    test_file = os.path.join(INPUT_DIR, 'test/test.csv')
    test_df = pd.read_csv(test_file)
    num_cols = [is_numeric_dtype(dtype) for dtype in test_df.dtypes]
    test_features = test_df.loc[:, num_cols]
    test_features = test_features.fillna(test_features.mean())
    test_features = mms.transform(test_features)
    preprocess(test_features)
    
    vals_indices, w_trains = adversarial_validation(train_features, test_features)
    w_trains = w_trains * 100.0 + 0.01 
    vals_x = train_features[vals_indices]
    vals_y = targets[vals_indices]

    # モデルの学習を行います.
    # X_train, X_test, y_train, y_test = train_features, vals_x, targets, vals_y
    clf = LogisticRegression()
    scoring = 'accuracy'
    # scoring = ''
    params = {
        'penalty': ['l1', 'l2'],
        'C': [.01, 0.5, 1.],
        'solver': ['saga', 'liblinear'],
        }
    model = GridSearchCV(
            clf,
            params,
            scoring=scoring,
            cv=5,
            )
    model.fit(X=train_features, y=targets, sample_weight=w_trains)


    pred = model.predict_proba(test_features)[:, 1]
    gt_file = os.path.join(INPUT_DIR, 'groundtruth/ground_truth.csv')
    gt_df = pd.read_csv(gt_file)
    gt_df = gt_df['default']

    auc_score = roc_auc_score(gt_df, pred)
    print(f'テストデータでのAUCスコア: {auc_score:.4f}')
    

    # 結果を出力します.
    output_file = os.path.join(INPUT_DIR, 'submission.csv')
    write_submission(test_df['id'], pred, output_file)


if __name__ == '__main__':
    main()