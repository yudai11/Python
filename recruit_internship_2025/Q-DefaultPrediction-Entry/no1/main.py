import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


INPUT_DIR = "../data"


def write_submission(id_list, pred, output_file):
    """予測結果を出力します."""
    df = pd.DataFrame({'id': id_list, 'default': pred})
    df.to_csv(output_file, index=False)


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

    # モデルの学習を行います.
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, targets, test_size=0.2, random_state=42
        )
    clf = LogisticRegression()
    scoring = 'accuracy'
    params = {
        'penalty': ['l1', 'l2'],
        'C': [.01, 1.],
        'solver': ['saga', 'liblinear'],
        }
    model = GridSearchCV(
            clf,
            params,
            scoring=scoring,
            cv=5,
            )
    model.fit(X=X_train, y=y_train)

    # 予測を行います.
    test_file = os.path.join(INPUT_DIR, 'test/test.csv')
    test_df = pd.read_csv(test_file)
    num_cols = [is_numeric_dtype(dtype) for dtype in test_df.dtypes]
    test_features = test_df.loc[:, num_cols]
    test_features = test_features.fillna(test_features.mean())
    test_features = mms.transform(test_features)
    pred = model.predict_proba(test_features)[:, 1]

    # 結果を出力します.
    output_file = os.path.join(INPUT_DIR, 'submission.csv')
    write_submission(test_df['id'], pred, output_file)


if __name__ == '__main__':
    main()
