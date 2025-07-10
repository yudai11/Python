# Q-DefaultPrediction-Entry

与えられたデータを使い、[問1](no1)に解答してください。

## 提出物

実行するコードを`no1`ディレクトリに格納し、GitHubにプッシュしてください。
各問のディレクトリをカレントディレクトリとして、`./run.sh`で起動できるようにしてください。

`run.sh`を実行すると、実行時ディレクトリからの相対パスで`../data`というディレクトリに`submission.csv`というファイルが結果として出力されるようにしてください。
以下で説明する入力用データは`../data`に入っているものとして実装してください。
データの読み込みと書き出しを行うPythonによる実装例は各問題ディレクトリに入っているので参考にしてください。
参考実装はPythonでの標準出力から書き出した例ですが、pandasなどのライブラリや、他プログラミング言語を利用した出力でも構いません。

尚、採点時は`../data`ディレクトリは採点用のデータに上書きされます。一時的にディスクに書き出したいデータがある場合は各問題のディレクトリ内など、別の場所に書き出すようにしてください。

ローカル環境で試しにコードを実行する際に、入出力パスを独自のものに設定したままコードを提出してしまうと、提出されたコードがうまく実行されませんので注意してください。
また、Windowsを使用している方は、入出力パスを「¥」区切りで記述したまま提出しないように注意してください。提出後のコードはLinux環境で実行されます。

## データ

以下のようなデータを使用します。

```
data
├── test
│   └── test.csv
└── train
    └── train.csv
```

- [data](data)にはローカル開発用のデータが入っています。ローカルでのモデル構築に利用してください。
- 採点時にはローカル開発用のデータとは別のデータが使われますが、形式は同一です。
- [data](data)にあるデータはそれぞれ1万行ですが、採点の際にはそれぞれ65万行ほどのデータが使われます。
- trainとtestはaddr_stateカラムの値を元に分割されています。

### `train`と`test`以下に含まれるファイル

各列がカンマで区切られたCSVファイルです。各列の意味と形式は以下のとおりです。

列名         | 意味                                        | 型   
----------- | ------------------------------------------- | --------------------------------
id | レコードに対する一意なID                                 | int
loan_amnt | 借り手が申請したローンの記載金額                    | float
term | ローンの支払い回数。値は月単位で、36または60のいずれか      | category
int_rate | ローンの金利                                      | float
installment | ローンが発生した場合に借り手が支払うべき毎月の支払い | float
grade | ローンのグレード                                     | category
emp_title | ローンを申請するときに借り手から提供された役職        | category
emp_length | 年単位の勤続年数。'< 1 year'は1年未満を、'10+ years'は10年以上を意味します　| category
home_ownership | 住宅所有権のステータス。                    　| category
annual_inc | 借り手の年収（自己申告）                          | float
verification_status | 収入、収入源が貸し手によって検証されたか否か | category
default | ローンの返済状況。1はデフォルト、0は返済完了を意味する        | int
purpose | ローンを借りる目的                                   | category
title | ローンを借りる目的（借り手が申告したもの）                 | category
addr_state | 借り手が住んでいるアメリカの州                       | category


以下にファイルのサンプルを示します。欠損が含まれるカラムも存在します。

`train.csv`

```
id,loan_amnt,term,int_rate,installment,grade,emp_title,emp_length,home_ownership,annual_inc,verification_status,default,purpose,title,addr_state
1,10000.00,36 months,11.44,329.48,B,Marketing,10+ years,RENT,117000.00,Not Verified,0,vacation,Vacation,PA
2,8000.00,36 months,11.99,265.68,B,Credit analyst,4 years,MORTGAGE,65000.00,Not Verified,1,debt_consolidation,Debt consolidation,IL
```

`test.csv`
```
id,loan_amnt,term,int_rate,installment,grade,emp_title,emp_length,home_ownership,annual_inc,verification_status,purpose,title,addr_state
1,7200.00,36 months,6.49,220.65,A,Client Advocate,6 years,RENT,54000.00,Not Verified,credit_card,Credit card refinancing,CA
2,24375.00,60 months,17.27,609.33,C,Destiny Management Inc.,9 years,MORTGAGE,55000.00,Verified,credit_card,Credit Card Refinance,TX
```

### `groundtruth/no1.csv`
確認用にサンプルデータに対する答え (`groundtruth/no1.csv`) が与えられています。
採点時には利用できません。

## 採点基準
採点用のテストケースを提出されたコードに対して入力し、得られた出力を用いてROC AUCを計算します。
テストケースに対して以下の`達成条件`を満たしているかを判定し、満たしている場合に`private scoreと点数`に応じて採点が行われます。
`private score`については[問1の問題文](no1)を参照してください
また上記採点基準とは別に、提出されたコードの内容（可読性、保守性など）を用いて参考材料とする場合があります。

### 達成条件

- 出力形式が正しい
- プロセスのメモリ使用量が`4GB`以下
- 処理時間が`1200`秒以内

### private scoreと点数

|private score|点数|
|--|--|
|0.70未満|0|
|0.70以上0.725未満|1|
|0.725以上0.750未満|2|
|0.750以上|3|

## フィードバック
採点時に各テストケースにおいて以下のフィードバック文のいずれかが返されます。

* `実行に成功し、結果ファイルがあることを確認しました。public score: xx`
  * 出力結果の`public score`が`xx`であることを表します
  * `public score`については[問1の問題文](no1)を参照してください
  * `private score`*以外*の達成条件は満たしていることを表します
* `コードの実行に失敗しました (Runtime Error)`
  * プロセスが正常に終了しなかったことを表します
* `コードの実行に失敗しました (Memory Limit Error)`
  * プロセスのメモリ使用量が達成条件の制限を超えたことを表します
* `コードの実行に失敗しました (Time Limit Error)`
  * 処理時間が達成条件の制限を超えたことを表します
* `コードの実行に失敗しました (Output Limit Error)`
  * プロセスの標準出力の制限を超えたことを表します
  * 標準出力の量を`10MiB`以下に減らして再提出してください
* `コードの実行に成功しましたが、結果ファイルが存在しませんでした。`
  * プロセスは正常終了したが、結果ファイルが出力されていないことを表します
* `コードの実行に成功しましたが、結果ファイルの形式が不正です。`
  * プロセスが正常終了して結果ファイルは存在するが、形式が不正であることを表します
* `サンプルのままの提出です`
  * 用意されたサンプルコードや解答例がそのまま提出されたことを表します
* `解答ファイルが提出されていません`
  * `run.sh`が提出されていないことを表します

その他、採点システム上の問題が生じた場合は下記のフィードバック文が返されます。お問い合わせ先にご連絡ください。

* `採点システムにエラーが発生しました`
  * システムのエラーです。お手数ですが本スキルチェックの案内メールに記載のお問い合わせ先にご連絡ください。



## ローカル実行方法

使用できる言語やライブラリについては [全体のREADME](../README.md) を参照してください。

ローカルでの動作確認は、Docker環境で行うことを推奨します。Docker環境を利用せずに動作確認する場合は、ファイルやディレクトリのパスの間違いやライブラリのバージョン違いに注意してください。

実行例（Pythonの場合）
```
（このREADMEがあるディレクトリをカレントディレクトリにした状態で）
docker build --platform linux/amd64 ../docker/python -t tester
docker run --rm -v $(pwd):/work -w /work tester sh -c "cd no1 && ./run.sh"
```

`data/submission.csv`として想定した結果が出力されることを確認してください。
