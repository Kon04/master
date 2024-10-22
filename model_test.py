#<概要>テストデータを用いて学習したモデルの性能評価を行うプログラム

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical

#諸変数の定義
num_classes = 5 #クラス数
model_path = './model/incptionv3_weights.h5' #評価するモデルのパス
test_data_path = 'test_img.npz'   #テストデータのパス
num_test = 750 #テストデータの枚数
#-------------------------------------------------------------

# npzファイルを読み込む(test)
test_data = np.load(test_data_path)

# 画像とラベルをそれぞれXとyに代入
X_test = test_data['img']
y_test = test_data['label']

#one-hotエンコーディング
# y_test = to_categorical(y_test, num_classes=num_classes)

#モデルと重みを復元
model = load_model(model_path)
 
#結果の表示
# accuracy = model.evaluate(X_test, y_test, verbose=0)
# print('test loss', accuracy[0])
# print('test accuracy', accuracy[1])

# テストデータでの予測
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1) #配列の最大値のインデックスを取得

print("y_test:", y_test[:10])  # 最初の10個を表示
print("y_pred_classes:", y_pred_classes[:10])  # 最初の10個を表示

# 混同行列の計算
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", cm)

# ヒートマップの作成
plt.figure(figsize=(8, 6))  # 図のサイズを指定
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.arange(num_classes), 
            yticklabels=np.arange(num_classes))

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# # 各クラスの精度の計算
# class_accuracy = cm.diagonal() / cm.sum(axis=1)

# # pandas DataFrameに変換して表形式で表示
# accuracy_df = pd.DataFrame({
#     'クラス': np.arange(len(class_accuracy)),
#     '精度': class_accuracy
# })

# print(accuracy_df)