# Iris Dataset Classification with scikit-learn

このプロジェクトは、scikit-learn ライブラリを使用して、アヤメ（Iris）のデータセットを分類する簡単な機械学習モデルを作成するものです。k 近傍法（K-Nearest Neighbors）アルゴリズムを用いて、がく片と花弁の寸法からアヤメの種類を予測します。

## アプリケーションの概要

このアプリケーションは、以下の機能を提供します。

- scikit-learn ライブラリを用いて、アヤメのデータセットをロードします。
- データを訓練データとテストデータに分割し、モデルを訓練します。
- 訓練されたモデルを用いて、テストデータのアヤメの種類を予測します。
- モデルの正解率を評価します。
- 予測結果を散布図で可視化します。
- 予測結果の詳細な表示と、予測の確率の表示を行います。

## 必要なライブラリ

このアプリケーションを実行するには、以下の Python ライブラリが必要です。

- **NumPy**: 数値計算ライブラリ
- **matplotlib**: グラフ描画ライブラリ
- **scikit-learn**: 機械学習ライブラリ

## インストール手順

以下の手順に従って、必要なライブラリをインストールしてください。

1.  Python がインストールされていることを確認してください。
2.  コマンドプロンプトまたはターミナルを開きます。
3.  以下のコマンドを実行して、必要なライブラリをインストールします。

    ```bash
    pip install numpy matplotlib scikit-learn
    ```

    または、Anaconda/Miniconda を使用している場合は、以下のコマンドを実行します。

    ```bash
    conda install numpy matplotlib scikit-learn
    ```

## 実行手順

1.  Jupyter Notebook を開きます。
2.  以下の Python コードを新しいセルにコピー＆ペーストします。

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # アヤメのデータセットをロード
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # データの分割（訓練データとテストデータ）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの作成と訓練（ここではk近傍法を使用）
    model = KNeighborsClassifier(n_neighbors=3)  # k=3のk近傍法
    model.fit(X_train, y_train)

    # テストデータでの予測
    y_pred = model.predict(X_test)

    # モデルの評価（正解率）
    accuracy = accuracy_score(y_test, y_pred)
    print(f"正解率: {accuracy}")

    # データの可視化（最初の2つの特徴量を使用）
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Predicted Iris Species')
    plt.show()

    # 予測結果の詳細な表示
    print("予測結果:")
    for i in range(len(y_test)):
        print(f"サンプル{i+1}: 予測={y_pred[i]}, 正解={y_test[i]}")

    # 予測の確率の表示
    y_proba = model.predict_proba(X_test)
    print("\n予測確率:")
    print(y_proba)
    ```

3.  セルを実行します（Shift + Enter）。
4.  結果が表示され、グラフが生成されます。

## 補足

- このコードは、k 近傍法（K-Nearest Neighbors）アルゴリズムを使用していますが、他の機械学習アルゴリズム（決定木、サポートベクターマシンなど）を試すこともできます。
- データの可視化では、最初の 2 つの特徴量（がく片の長さと幅）を使用していますが、他の特徴量の組み合わせを試すこともできます。
- 必要に応じて、コードを修正して、モデルのパラメータを調整したり、他の評価指標を使用したりすることができます。
