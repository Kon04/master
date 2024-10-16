#<概要>指定したディレクトリにあるファイルを任意の割合で分割するプログラム
#今回はデータセットを訓練データとテストデータに分ける

import os
import shutil
import random

#エントリーポイント
def main():
    #諸変数の定義
    original_dir = "C:/Kon/master_reserch/train_dataset" #分割元のデータセットフォルダーのパス
    base_dir = "C:/Kon/master_reserch/train_dataset2" #分割後のデータを格納するフォルダのパス
    train_size = 0.8 #訓練データの割合
    
    #データセット分割
    image_dir_train_test_sprit(original_dir, base_dir, train_size)

#データセットを訓練データとテストデータに分割する関数
def image_dir_train_test_sprit(original_dir, base_dir, train_size=0.8):
    '''
    画像データをトレインデータとテストデータにシャッフルして分割します。フォルダもなければ作成します。

    parameter
    ------------
    original_dir: str
      オリジナルデータフォルダのパス その下に各クラスのフォルダがある
    base_dir: str
      分けたデータを格納するフォルダのパス そこにフォルダが作られます
    train_size: float
      トレインデータの割合
    '''
    try: #try関数は特定のエラーに対しての例外処理を指定する
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir) #'original_dir'内のすべてのファイルとディレクトリのリストを取得
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))] #取得したリスト内のディレクトリのみをリストに保存
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists] #サブディレクトリまでのパスを生成

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    try:
        train_dir = os.path.join(base_dir, 'train') #訓練データ保存用フォルダーのパスを生成
        os.mkdir(train_dir)
    except FileExistsError:
        print(train_dir + "は作成済み")

    try:
        validation_dir = os.path.join(base_dir, 'test') #テストデータ保存用フォルダーのパスを生成
        os.mkdir(validation_dir)
    except FileExistsError:
        print(validation_dir + "は作成済み")

    #クラスフォルダの作成
    train_dir_path_lists = []
    val_dir_path_lists = []
    for D in dir_lists:
        train_class_dir_path = os.path.join(train_dir, D) #訓練データのクラスフォルダのパスを生成
        try:
            os.mkdir(train_class_dir_path)
        except FileExistsError:
            print(train_class_dir_path + "は作成済み")
        train_dir_path_lists += [train_class_dir_path] 
        
        val_class_dir_path = os.path.join(validation_dir, D) #テストデータのクラスフォルダのパスを生成
        try:
            os.mkdir(val_class_dir_path)
        except FileExistsError:
            print(val_class_dir_path + "は作成済み")
        val_dir_path_lists += [val_class_dir_path]


    #元データをシャッフルしたものを上で作ったフォルダにコピー
    #ファイル名を取得してシャッフル
    for i,path in enumerate(original_dir_path): #enumerate関数でリストの要素(path)とそのインデックス(i)を取得(今回はクラスフォルダー)
        files_class = os.listdir(path) #'path'で指定されたディレクトリの要素のリストを取得
        random.shuffle(files_class) #files_class内の要素をランダムにシャッフル
        # 分割地点のインデックスを取得
        num_bunkatu = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for fname in files_class[:num_bunkatu]:
            src = os.path.join(path, fname) #コピー元のファイルパスを生成
            dst = os.path.join(train_dir_path_lists[i], fname) #コピー先のファイルパスを生成
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for fname in files_class[num_bunkatu:]:
            src = os.path.join(path, fname)
            dst = os.path.join(val_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")

if __name__ == "__main__":
    main()
