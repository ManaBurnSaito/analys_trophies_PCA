# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA #主成分分析

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import os
import datetime
import re
import shutil

# %%
foil_file = "../LTR_fall.csv" #scryfallのcsvデータ
name_list = "../name_list.tsv" #カード名とカード画像の紐づけ
css_file = "ana.css" #html出力に使用するcss
set_flname = "LTR" #セット名

# %%
os.chdir(os.path.dirname(os.path.abspath('__file__'))) #カレントリディレクト移動、ファイルと同じ場所へ
inputdir = ''#作る場所のフォルダ
d_today = str(datetime.date.today())
outputdir = os.path.join(inputdir, d_today)


# %%
#csv/tsvファイルの読み込み。セット毎に書き換え
csv_file_path = "game_data_public.LTR.PremierDraft.csv" #public_dateのcsvを使用する時
#csv_file_path = "../trophiesROG/trop_resu.tsv" #自前のtsvを使用する時

# %%
#調べたいアーキカラー。好みで書き換え
#array_color = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
array_color = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG","UBR"]
#array_color = ["WU", "WG", "UB", "BR", "RG","WUB","WUG","WRG","UBR","BRG"]###SNC###

# %%
#除外する土地。セットにより特殊土地を書き換え。
remove_lands_array = [
    'Island',
    'Island.1',
    'Island.2',
    'Island.3',
    'Island.4',
    'Island.5',
    'Island.6',
    'Island.7',
    'Island.8',
    'Island.9',
    'Island.10',
    'Swamp',
    'Swamp.1',
    'Swamp.2',
    'Swamp.3',
    'Swamp.4',
    'Swamp.5',
    'Swamp.6',
    'Swamp.7',
    'Swamp.8',
    'Swamp.9',
    'Swamp.10',
    'Forest',
    'Forest.1',
    'Forest.2',
    'Forest.3',
    'Forest.4',
    'Forest.5',
    'Forest.6',
    'Forest.7',
    'Forest.8',
    'Forest.9',
    'Forest.10',
    'Mountain',
    'Mountain.1',
    'Mountain.2',
    'Mountain.3',
    'Mountain.4',
    'Mountain.5',
    'Mountain.6',
    'Mountain.7',
    'Mountain.8',
    'Mountain.9',
    'Mountain.10',
    'Plains',
    'Plains.1',
    'Plains.2',
    'Plains.3',
    'Plains.4',
    'Plains.5',
    'Plains.6',
    'Plains.7',
    'Plains.8',
    'Plains.9',
    'Plains.10',
    'Barad-dûr',
    'Great Hall of the Citadel',
    'Minas Tirith',
    'Mines of Moria',
    'Mount Doom',
    'Rivendell',
    'Shire Terrace',
    'The Grey Havens',
    'The Shire',
    'Unnamed: 2',
    ]

# %%
#除外するdfのカラム。自前のtsvやpublic_dataの仕様変更により書き換えが発生する必要があるかもしれない。
remove_calam_array = [
    'wins',
    'losses',
    'start_rank',
    'end_rank',
    #'colors',
    'aggregate_id',
    'deck_index',
    'time',
    'has_draft'
    ###public_date###
    'expansion',
    'event_type',
    #'draft_id',
    'draft_time',
    'build_index',
    'match_number',
    'game_number',
    'rank',
    'opp_rank',
    #'main_colors',
    'splash_colors',
    'on_play',
    'num_mulligans',
    'opp_num_mulligans',
    'opp_colors',
    'num_turns',
    'won'
    ###public_date###
    ]

# %%
def main():
    df_trans_obj = DataFrameTransformer(csv_file_path,remove_lands_array,remove_calam_array)
    generator = PCA_color_output(df=df_trans_obj.df,df_id=df_trans_obj.df_id,outputdir=outputdir, color_array=array_color,remove_calam_array=remove_calam_array,foil_file=foil_file,name_list=name_list,css_file=css_file,num=3)

# %%
class DataFrameTransformer:
    """
    データフレームの前処理

    Parameters:
        csv_file_path (pandas.DataFrame):前処理を行うデータフレーム
        ary_lands (list): dfから除外する土地カードなどの配列
        ary_calam (list): dfから除外するカード以外のカラム(ドラフトID等)

    """
    def __init__(self, csv_file_path, rem_lands_ary, rem_calam_ary) -> None:
        self.file_extension = None
        self.df = self.read_file_to_df(csv_file_path)
        self.df_color = None
        self.df_id = None
        self.ary_lands = rem_lands_ary #dfから除外する土地カード等
        self.ary_calam = rem_calam_ary #dfから除外するカード以外のカラム(ドラフトID等)
        
        if self.file_extension  == "csv":
            self.df = self.rename_df(self.df)
            self.df_id = self.public_date_dfmolding(self.df)

        if self.file_extension  == "tsv":
            self.df_id = self.dfmolding(self.df)

        self.df = self.remove_calam(self.df_id,self.ary_calam)



    #def main(self):
    #    self.df = self.dfmolding(df)

    def read_file_to_df(self, csv_file_path) -> pd.DataFrame:
        # ファイルの拡張子を取得
        file_extension = csv_file_path.split(".")[-1]

        # TSVファイルの場合
        if file_extension == "tsv":
            self.file_extension = "tsv"
            df = pd.read_table(csv_file_path, encoding='utf-8-sig')
            # タッチカラーの小文字削除
            df['colors'] = df['colors'].apply(lambda x: re.sub(r'[a-z]', '', x))

        # CSVファイルの場合
        elif file_extension == "csv":
            self.file_extension = "csv"
            # 分割する行数を指定する
            chunksize = 100000
            # 空のリストを作成して、分割したデータを一時的に保存する
            chunks = []
            
            # CSVファイルを分割して読み込み、一時的にリストに保存する
            for chunk in pd.read_csv(csv_file_path, encoding='utf-8-sig', chunksize=chunksize):
                chunks.append(chunk)
            
            # すべてのデータを結合して、一つのデータフレームに戻す
            df = pd.concat(chunks)

        else:
            raise ValueError("Unsupported file format. Only 'tsv' and 'csv' formats are supported.")
        
        return df


    #public_dateを読み込んだdfの成形
    def public_date_dfmolding(self,df) -> pd.DataFrame: 
        df = self.df_groupby(df)
        df = self.deck_del(df)
        df = self.del_columns_name(df)
        #df = self.remove_calam(df,self.ary_calam)
        df = self.remove_calam_lands(df,self.ary_lands)
        return df

    #自前のtsvを読み込んだdfの成形
    def dfmolding(self,df) -> pd.DataFrame: 
        #df = self.remove_calam(df,self.ary_calam)
        df = self.remove_calam_lands(df,self.ary_lands)
        return df

    def deck_del(self,df):
        #######deck_の列だけ抽出する#########
        # 対象となる文字列
        character = 'deck_'
        # 対象文字列を含む列名を取得
        column_inc_specific_char = [column for column in df.columns if character in column]

        # 'colors'カラムも追加
        if 'colors' in df.columns:
            column_inc_specific_char.append('colors')

        if 'draft_id' in df.columns:
            column_inc_specific_char.append('draft_id')

        # 取得した列名のみのデータフレーム
        df = df[column_inc_specific_char]
        return df

    #指定したカラムの削除・start_rank等のカード名ではないカラムの削除
    def remove_calam(self,df,remove_array):
        to_remove_word_calam = remove_array
        df = df[df.columns.difference(to_remove_word_calam)]
        return df

    #指定したカラムの削除・主に土地カードの削除    
    def remove_calam_lands(self,df,array):
        df = df[df.columns.difference(array)]
        return df
    
    def df_groupby(self,df):
        df = df[df['won']]
        df = df.groupby(['draft_id','colors'], as_index=False).sum()
        df = df.query('won == 7')
        return df
    
    def del_columns_name(self,df):
        df.columns = df.columns.str.replace('deck_', '')
        return df

    def flt_main_color(self, color):
        self.df_color = self.df[self.df['colors'] == color]
        self.df_color = self.df_color.drop(columns=['colors'])

    def rename_df(self,df):
        return df.rename(columns={'main_colors':'colors'})

# %%
class PCA_color_output:
    """
    PCAをして二次元にし非階層型クラスタリングで二次元の散布図を作る。

    Parameters:
        df (pandas.DataFrame):調べたいDF
        outputdir(str):出力ディレクトリのパス
        color (str): 調べたいアーキカラー["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
    """
    def __init__(self, df, df_id, outputdir, color_array,remove_calam_array,foil_file,name_list,css_file,num):
        self.df = df
        self.df_id = df_id
        self.df_cls = None
        self.df_color = None
        self.df_color_cls = None
        self.df_color_id = None


        self.ALL_color_pca(df,10)
        self.color_df_html(num)
        self.array_color_one_color_pca_doing(color_array,num)


    #指定したカラムの削除・主に土地カードの削除    
    def remove_calam_lands(self,df,array):
        df = df[df.columns.difference(array)]
        return df
        #df = self.remove_calam(df,self.ary_calam)


    #カラー振り分けなしの全アーキタイプのクラスタ振り分け
    #num数のクラスターを作成する
    def ALL_color_pca(self,df,num:int=10):
        df_A = self.remove_calam_lands(df,remove_calam_array)
        df_A = df_A.drop('colors', axis=1)
        df_A = df_A.drop('draft_id', axis=1)
        pca = PCA(n_components=2, whiten=False)
        pca.fit(df_A)
        feature = pca.transform(df_A)
        feature.shape
        data2 = feature
        km = KMeans(n_clusters=num, init='k-means++', n_init=10)

        df_A["cls"] = km.fit_predict(df_A)
        self.df_cls = df_A

        ax = plt.figure(figsize=(18,18)).gca()

        fig_color = ["r", "g", "c", "m","grey","tomato","gold","purple","pink","tan","lime","#1f77b4","#ff7f0e","#2ca02c","#8c564b","#17becf","#000080","#228B22","#CD5C5C","#5F9EA0"]

        sns.scatterplot(data=df_A, x=0, y=1, hue='cls',palette='deep' ,s=100)

        for i in range(data2.shape[0]):
            plt.scatter(data2[i,0], data2[i,1],c=fig_color[int(km.labels_[i])])
            if i % 20 == 0:
                #plt.text(data2[i,0], data2[i,1],df_Q.iat[i,0])#ドラフトIDつき
                #plt.text(data2[i,0], data2[i,1],df.iat[i,-1])#cls クラスタ番号
                plt.text(data2[i,0], data2[i,1],df['colors'].iloc[i])#アーキカラー
        plt.title("ALL_Archetype_PCA",fontsize='xx-large')


        if not os.path.exists(f'{outputdir}'):
            os.mkdir(f'{outputdir}')

        if not os.path.exists(f'{outputdir}/PCA_G_ALL'):
            os.mkdir(f'{outputdir}/PCA_G_ALL')

        plt.savefig(f'{outputdir}/PCA_G_ALL/ALL_Archetype_PCA.png',facecolor="white")

        plt.show()
        return df

    def array_color_one_color_pca_doing(self,color_array:list,num:int=3):
        for i in color_array:
            self.one_color_pca(i,num)
            self.one_color_df_html(foil_file,num,i)
            self.color_df_html_mean(i)
            self.make_css(css_file,num)


    def color_df_html(self,num):
        pca_num = self.df_cls['cls'].max()
        for ran in range(pca_num + 1):
            arryhtml = []

            df_q_g = self.df_cls[self.df_cls['cls'] == ran]
            df_q_g = df_q_g.drop('cls', axis=1)
            df_q_g_m = df_q_g.mean()
            dfm = pd.DataFrame(df_q_g_m)

            dfm['name'] = dfm.index
            print(dfm.head)


            nadf1 = pd.read_table(name_list)
            nadf = pd.read_csv(foil_file)
            nadf['name'] = nadf['name'].str.split(' // ').str[0]

            dfm_join = pd.merge(nadf,nadf1,on = "name",how = "outer")
            dfm_j = pd.merge(dfm,dfm_join,on = "name",how = "outer")

            df_test = dfm_j[dfm_j.iloc[:, 0]/7 > 0.2]

            array = df_test.cmc.drop_duplicates()
            arr = array.sort_values()
            print(arr)

            #cost
            arryhtml.append(f'<html><head><title>{ran}</title><link rel="stylesheet" type="text/css" href="ana.css"><meta charset=”UTF-8″></head><body><A>{set_flname} cls_PCA#{ran}</A><br />')

            firstLoopCMC = True
            for cm in arr:
                cma = cm
                df_ht = df_test.query('cmc == @cma')
                df_ht = df_ht.sort_values(0, ascending=False)

                arryhtml.append(f'<div class="box1"><C>{cm}</C>')
                #firstcard
                firstLoop = True
                for ht,row in df_ht.iterrows():
                    if firstLoop:#かきわけ
                        firstLoop = False
                        arryhtml.append(f'<div class=firstcard><img src="../../../img/{row.img}" width="160"/><AC>{round(row[0]*100/7,1)}%</AC></div>')
                        #nextcard
                    else:
                        arryhtml.append(f'<div class=NEXTcard><img src="../../../img/{row.img}" width="160"/><AC>{round(row[0]*100/7,1)}%</AC></div>')
                arryhtml.append(f'</div>')    


            arryhtml.append('</body></html>')
            arryhtmlsave = np.array(arryhtml)

            if not os.path.exists(f'{outputdir}'):
                os.mkdir(f'{outputdir}')

            if not os.path.exists(f'{outputdir}/PCA'):
                os.mkdir(f'{outputdir}/PCA')

            if not os.path.exists(f'{outputdir}/PCA/{num}'):
                os.mkdir(f'{outputdir}/PCA/{num}')

            np.savetxt(f'{outputdir}/PCA_G_ALL/{ran}.html',arryhtmlsave,'%s')




    def one_color_pca(self,chi_color,num):#7にまとめたdf
        df_color_id = self.df_id[self.df_id['colors'] == chi_color]
        df_A = df_color_id.drop(columns=['colors'])
        df_A = df_A.drop('draft_id', axis=1)
        pca = PCA(n_components=2, whiten=False)
        feature = pca.fit_transform(df_A)

        data2 = feature
        km = KMeans(n_clusters=num, init='k-means++', n_init=10) # n種類のグループに分ける

        df_d = pd.DataFrame(data2)
        df_d["cls"] = km.fit_predict(df_A)
        df_A["cls"] = df_d["cls"].values

        ax = plt.figure(figsize=(20,20)).gca()

        sns.scatterplot(data=df_d, x=0, y=1, hue='cls',palette='deep' ,s=100)

        # テキストを記録するリストを作成
        text_to_save = []

        for i in range(data2.shape[0]):
            #plt.scatter(data2[i,0], data2[i,1],c=color[int(km.labels_[i])])
            if i % 100 == 0:
                #plt.text(data2[i,0], data2[i,1],df2.iat[i, 1])
                plt.text(data2[i,0], data2[i,1],df_color_id['draft_id'].iloc[i])#ドラフトIDつき
                x, y = data2[i, 0], data2[i, 1]
                # テキストを記録するリストに追加
                text_to_save.append(f"{df_color_id['draft_id'].iloc[i]}")
                #plt.text(data2[i,0], data2[i,1],df_Q_del.iat[i,-1])
                #plt.text(data2[i,0], data2[i,1],df_A.loc[i, "cls"])#クラス

        plt.title(chi_color,fontsize='xx-large')
        plt.legend(fontsize=16)

        if not os.path.exists(f'{outputdir}'):
            os.mkdir(f'{outputdir}')

        if not os.path.exists(f'{outputdir}/PCA'):
            os.mkdir(f'{outputdir}/PCA')

        if not os.path.exists(f'{outputdir}/PCA/{num}'):
            os.mkdir(f'{outputdir}/PCA/{num}')

        plt.savefig(f'{outputdir}/PCA/{num}/{chi_color}.png',facecolor="white")
        #plt.show()

        # テキストファイルに保存
        with open(f"{outputdir}/PCA/{num}/{chi_color}_draftID.txt", "w" ,encoding='utf-8-sig') as file:
            for text in text_to_save:
                file.write(text + "\n")

        self.df_color_cls = df_A


    def one_color_df_html(self,foil_file,pca_num,color):
            for ran in range(pca_num):
                arryhtml = []
                df_q_g = self.df_color_cls[self.df_color_cls['cls'] == ran]
                df_q_g = df_q_g.drop('cls', axis=1)
                df_q_g_m = df_q_g.mean()
                dfm = pd.DataFrame(df_q_g_m)

                dfm['name'] = dfm.index

                nadf1 = pd.read_table(name_list)
                nadf = pd.read_csv(foil_file)
                nadf['name'] = nadf['name'].str.split(' // ').str[0]

                dfm_join = pd.merge(nadf,nadf1,on = "name",how = "outer")
                dfm_j = pd.merge(dfm,dfm_join,on = "name",how = "outer")

                df_test = dfm_j[dfm_j.iloc[:, 0]/7 > 0.2]

                array = df_test.cmc.drop_duplicates()
                arr = array.sort_values()
                print(arr)

                #cost
                arryhtml.append(f'<html><head><title>{color}</title><link rel="stylesheet" type="text/css" href="ana.css"><meta charset=”UTF-8″></head><body><A>{set_flname} {color}_PCA#{ran}</A><br />')

                firstLoopCMCA = True
                for cm in arr:
                    cma = cm
                    df_ht = df_test.query('cmc == @cma')
                    df_ht = df_ht.sort_values(0, ascending=False)

                    arryhtml.append(f'<div class="box1"><C>{cm}</C>')
                    #firstcard
                    firstLoop = True
                    for ht,row in df_ht.iterrows():
                        if firstLoop:#かきわけ
                            firstLoop = False
                            arryhtml.append(f'<div class=firstcard><img src="../../../../img/{row.img}" width="160"/><AC>{round(row[0]*100/7,1)}%</AC></div>')
                            #nextcard
                        else:
                            arryhtml.append(f'<div class=NEXTcard><img src="../../../../img/{row.img}" width="160"/><AC>{round(row[0]*100/7,1)}%</AC></div>')
                    arryhtml.append(f'</div>')    



                arryhtml.append('</body></html>')
                arryhtmlsave = np.array(arryhtml)

                if not os.path.exists(f'{outputdir}/PCA/{pca_num}'):
                    os.mkdir(f'{outputdir}/PCA/{pca_num}')


                np.savetxt(f'{outputdir}/PCA/{pca_num}/{color}_{ran}.html',arryhtmlsave,'%s')

    def make_css(self,css_file,pca_num):
        shutil.copy(css_file,f'{outputdir}/PCA/{pca_num}')
        shutil.copy(css_file,f'{outputdir}/PCA_G_ALL')
        shutil.copy(css_file,f'{outputdir}/color_mean')

    def color_df_html_mean(self,color):
        arryhtml = []
        df_A = self.df_id[self.df_id['colors'] == color]
        df_A = df_A.drop('colors', axis=1)
        df_A = df_A.drop('draft_id', axis=1)
        df_q_g_m = df_A.mean()
        dfm = pd.DataFrame(df_q_g_m)
        dfm['name'] = dfm.index
        #dfm['name'] = dfm['name'].str[5:]
        #print(dfm.head)
        nadf1 = pd.read_table(name_list)
        nadf = pd.read_csv(foil_file)
        print(nadf.head(5))
        nadf['name'] = nadf['name'].str.split(' // ').str[0]
        print(nadf.head(5))
        dfm_join = pd.merge(nadf,nadf1,on = "name",how = "outer")
        dfm_j = pd.merge(dfm,dfm_join,on = "name",how = "outer")
        dfm_j.iloc[:, 0] = dfm_j.iloc[:, 0] / 7
        df_test = dfm_j[dfm_j.iloc[:, 0] > 0.2]
        array = df_test.cmc.drop_duplicates()
        arr = array.sort_values()
        print(arr)
        
        #cost
        arryhtml.append(f'<html><head><title>{color}</title><link rel="stylesheet" type="text/css" href="ana.css"><meta charset=”UTF-8″></head><body><A>{set_flname} {color}</A><br />')
        firstLoopCMC = True
        for cm in arr:
            cma = cm
            df_ht = df_test.query('cmc == @cma')
            df_ht = df_ht.sort_values(0, ascending=False)
            arryhtml.append(f'<div class="box1"><C>{cm}</C>')
            #firstcard
            firstLoop = True
            for ht,row in df_ht.iterrows():
                if firstLoop:#かきわけ
                    firstLoop = False
                    arryhtml.append(f'<div class=firstcard><img src="../../../img/{row.img}" width="160"/><AC>{round(row[0]*100,1)}%</AC></div>')
                    #nextcard
                else:
                    arryhtml.append(f'<div class=NEXTcard><img src="../../../img/{row.img}" width="160"/><AC>{round(row[0]*100,1)}%</AC></div>')
            arryhtml.append(f'</div>')    


        arryhtml.append('</body></html>')
        arryhtmlsave = np.array(arryhtml)
        if not os.path.exists(f'{outputdir}'):
            os.mkdir(f'{outputdir}')
        if not os.path.exists(f'{outputdir}/color_mean/'):
            os.mkdir(f'{outputdir}/color_mean/')
        np.savetxt(f'{outputdir}/color_mean/{color}.html',arryhtmlsave,'%s')



# %%
if __name__ == "__main__":
    main()

# %%
#htmlを画像にする
import glob
import imgkit

def output_img(outputdir):
    files = glob.glob(outputdir + "/*.html")
 
    options = {
        'enable-local-file-access': None #ローカルファイルの読み込み許可
    }

    for file in files:
        print(file)
        basename_without_ext = os.path.splitext(os.path.basename(file))[0]
        print(basename_without_ext)
        if not os.path.exists(outputdir + '/img'):
            os.mkdir(outputdir + '/img')
        imgkit.from_file(file, outputdir+"/img/"+basename_without_ext + '.jpg', options=options)

# %%
pca_num = 3
output_img(f'{outputdir}/PCA/{pca_num}')
output_img(f'{outputdir}/PCA_G_ALL')
output_img(f'{outputdir}/color_mean')

# %%



