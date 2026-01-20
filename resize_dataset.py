#<概要>画像のサイズをリサイズするプログラム

import os
import glob
from PIL import Image

#エントリーポイント
def main():
    #諸変数の定義
    original_dir = 'C:\\Kon\\master_reserch\\datasets\\simulation'  #リサイズ元の画像フォルダのパスを指定
    save_dir = 'C:\\Kon\\master_reserch' #リサイズした画像を保存するディレクトリを指定
    save_name = "simulation_resized299" #保存用のフォルダー名
    resize_size = 299 #リサイズ後の画像サイズ

    #保存先の作成
    save_dir_plus = os.path.join(save_dir, save_name)
    os.makedirs(save_dir_plus, exist_ok=True)

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir) #'original_dir'内のすべてのファイルとディレクトリのリストを取得
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))] #取得したリスト内のディレクトリのみをリストに保存

    for d in dir_lists:
        #保存先の作成
        save_dir_plus = os.path.join(save_dir, save_name, d)
        os.makedirs(save_dir_plus, exist_ok=True)
        
        #リサイズする関数へ
        resize_files(d, original_dir, save_dir_plus, resize_size)

#画像をリサイズする関数
def resize_files(dir_lists, original_dir, save_dir, resize_size):
    print('a')
    #画像ファイルのパスを取得
    original_dir_path = os.path.join(original_dir, dir_lists) #サブディレクトリまでのパスを生成
    files_path = os.path.join(original_dir_path, '*')
    files = glob.glob(files_path)

    #処理&保存
    for f in files:
        print('b')
        root, ext = os.path.splitext(f) #ファイルパスを拡張子とそれ以外に分割(extに拡張子が格納される)
        if ext.upper() in ['.JPG', '.PNG', '.JPEG']: #拡張子が画像ファイルのものか判別
            
            #リサイズ処理
            img = Image.open(f)
            size = (resize_size, resize_size)
            img_resize = img.resize(size)
            
            #リサイズした画像の保存
            file_name = os.path.basename(root) #パスのファイル名部分だけを取得
            save_path = os.path.join(save_dir, file_name + ext)
            img_resize.save(save_path)
            print('c')
            
if __name__ == "__main__":
    main()
