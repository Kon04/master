###<概要>データセットを取得し、学習用の処理を加えnpzファイルを生成

# 必要なものをimport
import tensorflow as tf
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

#諸変数の定義
img_size = 299

# ディレクトリの指定
train_dir = 'C:\\Kon\\master_reserch\\tensorflow\\train_dataset\\train'
test_dir = 'C:\\Kon\\master_reserch\\tensorflow\\train_dataset\\test'

# カテゴリネームと番号を対応させた辞書を作成（ラベルは数値で保存します）
label_dict = {
    '50ml' : 0,
    '100ml' : 1,
    '150ml' : 2,
    '200ml' : 3,  
    '250ml' : 4}

# 画像とラベルを格納する空のリストを作る
Images_train = []
Labels_train = []

# 指定したディレクトリの画像パスを全て取得して処理していく
for image_path in glob.glob(train_dir + '/*' + '/*'):
    label_name = image_path.split('\\')[-2]

    # ラベル番号の取得
    label = label_dict[label_name]
    print(label)
  
    # 画像の読み込み
    img = cv2.imread(image_path)

    # RGB形式に変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # サイズ変換
    img_resize = cv2.resize(img_rgb, (img_size,img_size))

    # ndarray型に変換
    numpy_img = np.array(img_resize)

    # 指定した形にリシェイプ
    numpy_img = numpy_img.reshape(img_size, img_size, 3)

    # float32型にして255で割ることで[0,1]スケールの正規化
    numpy_img = numpy_img.astype('float32')
    numpy_img = numpy_img / 255

    # 画像とラベルをリストに格納
    Images_train.append(numpy_img)
    Labels_train.append(label)

#npzファイル作成
# 第1引数はファイル名、第2引数以降は [キー名 = 配列]の形で書く
np.savez('train_img.npz', img = Images_train, label = Labels_train)

print('train終了')

#----------------------------------------------------
#testについて処理
# 画像とラベルを格納する空のリストを作る
Images_test = []
Labels_test = []

# 指定したディレクトリの画像パスを全て取得して処理していく
for image_path in glob.glob(test_dir + '/*' + '/*'):
    label_name = image_path.split('\\')[-2] 

    # ラベル番号の取得
    label = label_dict[label_name]
    print(label)
  
    # 画像の読み込み
    img = cv2.imread(image_path)

    # RGB形式に変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # サイズ変換
    img_resize = cv2.resize(img_rgb, (img_size,img_size))

    # ndarray型に変換
    numpy_img = np.array(img_resize)

    # 指定した形にリシェイプ
    numpy_img = numpy_img.reshape(img_size, img_size, 3)

    # float32型にして255で割ることで[0,1]スケールの正規化
    numpy_img = numpy_img.astype('float32')
    numpy_img = numpy_img / 255

    # 画像とラベルをリストに格納
    Images_test.append(numpy_img)
    Labels_test.append(label)

#npzファイル作成
# 第1引数はファイル名、第2引数以降は [キー名 = 配列]の形で書く
np.savez('test_img.npz', img = Images_test, label = Labels_test)

print('test終了')