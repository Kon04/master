#<概要>テストデータを用いて学習したモデルの性能評価を行うプログラム

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

#諸変数の定義
num_classes = 5 #クラス数
model_path = './model/inceptionv3_weights_real_simulation4200.h5' #評価するモデルのパス
test_data_path = 'test_unknown_img.npz'   #テストデータのパス
save_name = 'incv3_real_simulation4200_confusion_matrix_unknown' #混合行列保存用のファイル名
num_test = 1250 #テストデータの枚数
confusion_flag = 1 #混合行列を表示するかのフラグ（1の時表示）
show_flag = 0 #分類を間違えた画像を表示するかのフラグ（1の時表示）
#-------------------------------------------------------------

# npzファイルを読み込む(test)h
test_data = np.load(test_data_path)

# 画像とラベルをそれぞれXとyに代入
X_test = test_data['img']
y_test = test_data['label']

#one-hotエンコーディング
y = to_categorical(y_test, num_classes=num_classes)

#モデルと重みを復元
model = load_model(model_path)
 
#結果の表示
accuracy = model.evaluate(X_test, y, verbose=0)
print('test loss', accuracy[0])
print('test accuracy', accuracy[1])

# テストデータでの予測
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1) #配列の最大値のインデックスを取得

#混合行列を表示
if(confusion_flag == 1):
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
    plt.savefig(save_name)

#予測を間違った画像を表示
if(show_flag == 1):
    # 間違って分類されたインデックスを取得
    incorrect_indices = np.where(y_pred_classes != y_test)[0]

    # 一ページに表示する画像の数
    images_per_page = 20
    num_pages = len(incorrect_indices) // images_per_page + (1 if len(incorrect_indices) % images_per_page != 0 else 0)

    # ページごとに画像を表示
    for page in range(num_pages):
        plt.figure(figsize=(30, 18))  # 1ページのサイズを設定（適宜調整）

        # 1ページに表示する画像のインデックス
        start_index = page * images_per_page
        end_index = min((page + 1) * images_per_page, len(incorrect_indices))

        # サブプロットを設定
        for i in range(start_index, end_index):
            index = incorrect_indices[i]
            plt.subplot(4, 5, i - start_index + 1)  # 4行×5列のサブプロットを設定

            # 画像を表示
            plt.imshow(X_test[index], cmap='gray')  # データがグレースケールの場合
            plt.title(f"True: {y_test[index]}, Pred: {y_pred_classes[index]}")
            plt.axis('off')  # 軸を非表示

       # ページごとに画像を保存
        page_filename = f"incorrect__images_page_{page + 1}.png"
        plt.tight_layout()  # レイアウトを自動調整
        plt.savefig(page_filename)  # 画像を保存
        plt.close()  # 現在の図を閉じてメモリを解放

        print(f"Page {page + 1} saved as {page_filename}")