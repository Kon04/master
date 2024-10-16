###<概要>任意の頻度で動画から画像を切り出すプログラム
import cv2
import numpy as np
import os

#諸変数の定義
i=0
count = 0
cpf = 10                #何フレーム毎に切り出すか
image_width = 299       #リサイズする画像の幅
image_heigh = 299       #リサイズする画像の高さ
laplacian_thr = 0     #ボケ画像判定をするときのスレッショルド
video_path = 'C:/Kon/master_reserch/vol_dataset/pra/P1140009.MOV'         #読み込む動画ファイルのパスを指定
save_path = 'C:/Kon/master_reserch/vol_dataset/pra/'  #保存先の親フォルダのパス
save_dir = '50ml'   #保存先の子フォルダの名前
save_name = 'pra50'   #保存する画像名
extension = '.jpg'  #画像ファイルの拡張子
save_file_num = 20 #保存する画像の数

#動画ファイル
cap = cv2.VideoCapture(video_path)

while (cap.isOpened()):
    ret, img = cap.read()
    
    #読み込み失敗時は終了
    if ret == False:
        print('Finished')
        break
    
    if count%cpf == 0:  #cpfフレームごとに処理
        print('aaa')
        
        #サイズを小さくする
        resize_frame = cv2.resize(img,(image_width,image_heigh))
        #cv2.imshow('Video', resize_frame)
        
        #画像がぶれていないか確認する
        laplacian = cv2.Laplacian(resize_frame, cv2.CV_64F)
        if ret and laplacian.var() >= laplacian_thr: #ピンぼけ判定がしきい値以上のもののみ出力

            #保存する画像のパス生成
            s_name = os.path.join(save_path, save_dir, f'{save_name}_{i}{extension}')
            print(s_name)
            
            #画像の保存
            write = cv2.imwrite(s_name, resize_frame)
            assert write, "保存に失敗"
            print('Save', save_name + '_' + str(i)) #確認用表示
            i += 1
        

        # キー入力を処理する
        key = cv2.waitKey(25)  # 25ミリ秒ごとに次のフレームを表示する
        if key & 0xFF == ord('q'):  # 'q'キーが押されたら終了する
            break
        
    if i == save_file_num:  #指定した保存する画像の数を満たしたら終了
        print("指定した数の画像を保存しました")
        break
    
    count = count + 1

cap.release()