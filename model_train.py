#<概要>inceptionv3の転移学習を行うプログラム
import tensorflow as tf
import numpy as np
import callbacks
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical

#諸変数の定義
base_model_name =InceptionV3 #転移学習元のネットワークを指定
num_classes = 5 #クラス数
optimizer =  Adam() #最適化手法
loss = 'categorical_crossentropy' #損失関数
metrics = 'accuracy' #評価関数
model_display_flag = 1 #モデルの層構造を表示するフラグ(1の時表示)
validation_split = 0.2 #検証データの割合
batch_size = 32 #ミニバッチサイズ 
freeze_layer = 249 #ファインチューニングで凍結させる層の数(ここで指定したn-1層までが凍結される)
epochs = 30 #エポック数
set_dir_name = 'inceptionv3_simulation' #テンソルボードのログ保存用ディレクトリの名前
train_data_path = 'real_simulation4200_train.npz' #訓練データのパス
test_data_path = 'real_simulation4200_test.npz'   #テストデータのパス
num_test = 930 #テストデータの枚数
full_con_ep = 15 #全結合層学習のエポック数
model_save_path = './model/incptionv3_weights_real_simulation4200.h5'

#---------------------------------------------------------
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

#次元追加（３次元→４次元）
dataset_test = dataset_test.batch(num_test, drop_remainder=True)

#転移学習元のネットワークをダウンロード
#"include_top=False"の場合全結合層を除いたネットーワークを取得
base_model = base_model_name(weights="imagenet", include_top=False)

#全結合層の追加
x = base_model.output
x = GlobalAveragePooling2D()(x)  # グローバル平均プーリング層を追加
predictions = Dense(num_classes, activation="softmax")(x) #出力層を追加

#新モデルの定義
model = Model(inputs=base_model.input, outputs=predictions)

#モデルの学習
#最初は追加した層部分のみ学習を行う
for layer in base_model.layers:
    layer.trainable = False

#モデルのコンパイル
model.compile(optimizer=RMSprop(),
              loss=loss,
              metrics=[metrics])

#モデルの層構造の表示
if model_display_flag == 1:
    model.summary()

#学習(全結合層のみ)
model.fit(dataset_train, epochs=full_con_ep, validation_data=val_dataset)

#結果の表示
accuracy = model.evaluate(dataset_test, verbose=0)
print('テスト結果(全結合層)',accuracy)

#ファインチューニング
#最初のfreeze_layar層は学習せず、freeze_layer層以降は学習させる
base_model.trainable = True

for layer in model.layers[freeze_layer:]: #スライス[start:stop]はstopを含まないことに注意
    layer.trainable = False

for layer in model.layers[:freeze_layer]:
    layer.trainable = True

#モデルの再コンパイル(layer.trainableの設定後に、必ずcompile)
model.compile(optimizer=Adam(),
              loss=loss,
              metrics=[metrics])

#モデルの層構造の表示
if model_display_flag == 1:
    model.summary()
    
#コールバックを指定
early_stopping = callbacks.early_stop()
reduce_lr = callbacks.reduce_learnrate()
logging = callbacks.make_tensorboard(set_dir_name)

#学習
model.fit(dataset_train, 
          epochs=epochs, 
          validation_data=val_dataset,
          callbacks = [early_stopping, reduce_lr, logging],
          verbose=1
          )

#モデルと重みの保存
model.save(model_save_path) 

#結果の表示
accuracy = model.evaluate(dataset_test, verbose=0)
print('test loss', accuracy[0])
print('test accuracy', accuracy[1])