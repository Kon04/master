# 必要なものをimport
import tensorflow as tf
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

train_data_path = 'train_img.npz' #訓練データのパス
test_data_path = 'test_img.npz'   #テストデータのパス
batch_size = 32
validation_split = 0.2
num_classes = 5

#データセットの取得
# npzファイルを読み込む(train)
train_data = np.load(train_data_path)

# 画像とラベルをそれぞれXとyに代入
X = train_data['img']
y = train_data['label']

#one-hotエンコーディング
y = to_categorical(y, num_classes=num_classes)

# X:画像データ y:ラベルデータをセットでdataset化
dataset_train = tf.data.Dataset.from_tensor_slices((X, y))

# データセットのサイズを取得
dataset_size = sum(1 for _ in dataset_train)

# バリデーションセットのサイズを計算
validation_size = int(validation_split * dataset_size)

# シャッフル
dataset_train = dataset_train.shuffle(buffer_size=len(X))

# 訓練セットとバリデーションセットに分割
dataset_train = dataset_train.skip(validation_size)
val_dataset = dataset_train.take(validation_size)

# ミニバッチ化
#（drop_remainder = Trueで端数切り捨て）
dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

# npzファイルを読み込む(test)
test_data = np.load(test_data_path)

# 画像とラベルをそれぞれXとyに代入
X = test_data['img']
y = test_data['label']

#one-hotエンコーディング
y = to_categorical(y, num_classes=num_classes)

# X:画像データ y:ラベルデータをセットでdataset化
dataset_test = tf.data.Dataset.from_tensor_slices((X, y))

# シャッフル
dataset_test = dataset_test.shuffle(buffer_size=len(X))

# 最初の要素を取得して形状を確認
count = 0
for images in dataset_test:
    count += 1

print(count)