# 必要なものをimport
import tensorflow as tf
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

train_data_path = 'train_img.npz' #訓練データのパス
test_data_path = 'test_img.npz'   #テストデータのパス

#データセットの取得
# npzファイルを読み込む(train)
train_data = np.load(train_data_path)

# 画像とラベルをそれぞれXとyに代入
X = train_data['img']
y = train_data['label']

# X:画像データ y:ラベルデータをセットでdataset化
dataset_train = tf.data.Dataset.from_tensor_slices((X, y))

# シャッフル
dataset_train = dataset_train.shuffle(buffer_size=len(X))
print(X.shape, y.shape)

# npzファイルを読み込む(test)
test_data = np.load(test_data_path)

# 画像とラベルをそれぞれXとyに代入
X = test_data['img']
y = test_data['label']

# X:画像データ y:ラベルデータをセットでdataset化
dataset_test = tf.data.Dataset.from_tensor_slices((X, y))

# シャッフル
dataset_test = dataset_test.shuffle(buffer_size=len(X))
print(X.shape, y.shape)
