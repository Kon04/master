#<概要>テストデータを用いて学習したモデルの性能評価を行うプログラム

import tensorflow as tf
import numpy as np
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
y_test = to_categorical(y_test, num_classes=num_classes)

print(X_test.shape)
print(y_test.shape)

# X:画像データ y:ラベルデータをセットでdataset化
dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# シャッフル
dataset_test = dataset_test.shuffle(buffer_size=len(X_test))

#次元追加（３次元→４次元）
dataset_test = dataset_test.batch(num_test, drop_remainder=True)

#モデルと重みを復元
model = load_model(model_path)
 
#結果の表示
accuracy = model.evaluate(X_test, y_test, verbose=0)
print('test loss', accuracy[0])
print('test accuracy', accuracy[1])