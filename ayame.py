# 必要なライブラリのインポート
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
