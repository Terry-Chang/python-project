#%%
"""
參考網頁
https://stackoverflow.com/questions/36028759/how-to-open-and-convert-sqlite-database-to-pandas-dataframe

從SQL的表 撈出 做成PD 再試整理
"""

import os
import time
import sqlite3
import numpy as np
import pandas as pd

#%%
# 確認檔案 和 路徑
# print(os.getcwd())
path=os.listdir('./db')
path[0]
path='./db'+'/'+path[0]

# Create your connection.
cnx = sqlite3.connect(path)
# cnx = sqlite3.connect(".\\df\\stockNo_2330.sqlite")

cursor=cnx.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type = "table"')
Tnamelist = cursor.fetchall()


Tdatall=[]
i=Tnamelist[0]
for i in Tnamelist:
    
    print(str(i[0]))
    
    print("SELECT * FROM "+str(i[0]))
    df = pd.read_sql_query(("SELECT * FROM "+str(i[0])), cnx)
    Tdatall.append(df)
    time.sleep(0.2)
    

cnx.close()
time.sleep(0.5)


dfall = pd.concat(Tdatall)
dfall = dfall.reset_index(drop=True)

# 清除不用的變數
del Tdatall, Tnamelist, df, i, cursor, path

#%%
# 改英文 colname
ch = pd.DataFrame(dfall.columns, columns=['ch'])
en = pd.DataFrame(['date', 'shares', 'amount', 'open', 'high', 'low', 'close', 'change', 'turnover'], columns=['en'])
ch_en = pd.concat([ch,en], axis=1)

del ch, en

dfall.columns = ch_en['en']

#%%
# 確認一下 型態  
# 兩種偷看型態法 很重要
dfall.info()
dfall.dtypes


# 先處理字串裡的 逗號問題
dfall.iloc[:,1:] = dfall.iloc[:,1:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))

# 再處理 .dtypes 的問題  dtypes => object 有些數學模型是不吃的
dff1 = dfall.iloc[:,0:1]
dff2 = dfall.iloc[:,1:]
dff2=dff2.astype('float32')

dff2.info()
dff2.dtypes


dfall = pd.concat([dff1, dff2], axis=1)


del dff1, dff2

#%%  先把星期幾做出來

# 淺複製(shallow copy)與深複製(deep copy)
# import copy 
# df_date=copy.deepcopy(DFall['日期'])

dfall.columns
type(dfall['date'][0])

df_date=pd.DataFrame(dfall['date']) # 經查驗此操作為深複制
df_date.iloc[0,0]=' 00/01/04'
df_date=pd.DataFrame(dfall['date'])


# df_date[['yy', 'mm', 'dd']]=df_date.日期.str.split('/', expand=True)
df_date[['yy', 'mm', 'dd']]=df_date["date"].str.split('/', expand=True)
df_date['yy'][0]

df_date['yy']=df_date['yy'].str.strip()



"""
# apply函數是`pandas`裡面所有函數中自由度最高的函數
# 可以自定義一個 Python function 並將 apply 函式套用到整個 DataFrame 之上

def gg(row):
    return int(row)+1911

df_date['newyy'] = df_date.yy.apply(gg)
"""

df_date['ADyear'] = df_date.yy.apply(lambda x: str(int(x)+1911))
# 透過 apply 函式
# 我們把一個匿名函式 lambda 套用到整個 df.Survived Series 之上
# 並以此建立一個新的 存活 欄位。

# 其中 lambda x:  為匿名函式的開頭 也稱 宣告
# 後面接要 return 的東西


df_date.head(5)
df_date['ADdate'] = df_date['ADyear'] + '-' + df_date['mm'] + '-' + df_date['dd']
df_date.head(5)

dfall = pd.merge(df_date,dfall, on='date')
dfall.dtypes

del df_date

#%%
# 接上 CSV.db => 美股台積資料 台灣加權指(大盤) 
"""
台股開盤時間是09:00-13:30（台北時間）
美國的開盤時間比較長09:30-16:00（紐約時間）
另外美國有實行夏令日光節約時間
因此夏令開盤時間是21:30-04:00（台北時間）
冬令開盤時間是22:30-05:00（台北時間）。


台北的時間比美國紐約州紐約快 12 小時
台北的星期一下午9:36 是
美國紐約州紐約的星期一上午9:36

ex
台北時間 4/28 AM 9:00 開盤
紐約時間 4/27 PM 9:00 

所以 台北4/28的盤  要參照 紐約 4/27的股市資料
"""
print(os.listdir('./csv'))

df_NYSE_TSM = pd.read_csv('./csv/NYSE_TSM.csv')
df_NYSE_TSM.info()

df_TWII = pd.read_csv('./csv/^TWII.csv')
df_TWII.info()

# 台積美股 先和 台灣大盤  合併
res1 = pd.merge(df_TWII.loc[:, ['Date','Close']], 
               df_NYSE_TSM.loc[:, ['Date','Close']],
               on=['Date'],
               suffixes=['_TWII','_NYSE'],
               how='left')

# 改名字的坑 超多的啦 之前試過好幾種  後續都有問題
res1 = res1.rename(columns={"Date": "ADdate"})


# 三張表 出來了 要合併  有些col是不要的
# 乾  官網的講說 懶到一個炸裂
res2 = pd.merge(dfall.loc[:, ['ADdate', 'close', 'change']],res1, 
                on=['ADdate'], how='left')



#%%
# KD 指標
"""
未成熟隨機值(RSV)：
(今日收盤價 - 最近9天的最低價) / (最近9天最高價 - 最近9天的最低價)

當日K值：前日K值 * (2/3) + 當日RSV值 * (1/3)
當日D值：前日D值 * (2/3) + 當日K值 * (1/3)
"""

import talib

# talib.abstract.STOCH(dfall)

dfall['k'], dfall['d'] = talib.STOCH(dfall['high'], 
                                     dfall['low'], 
                                     dfall['close'])

#%%
# 移動平均
# i=list(dfall.columns).index('close')

# dfall['MA5']=dfall.iloc[:,i].rolling(5).mean()
# dfall['MA10']=dfall.iloc[:,i].rolling(10).mean()
# dfall['MA15']=dfall.iloc[:,i].rolling(15).mean()
# dfall['MA30']=dfall.iloc[:,i].rolling(30).mean()

# del i

#%%
# 資料還沒撈完  先做看看熱圖

# 定義一個函數 做corr 然後畫圖
def hotmapp (dff):
    print(res2.info())
    print(res2.dtypes)
    
    import matplotlib.pyplot as plt
    
    corr_arr = dff.corr(method ='pearson')
    
    plt.subplots(figsize=(10, 10)) # 设置画面大小
    
    import seaborn as sns
    sns.heatmap(corr_arr, annot=True, vmax=1, square=True, cmap="Blues")
    
    plt.show()
    


# =============================================================================
# 
# import matplotlib.pyplot as plt
# 
# dff=dfall.iloc[:,11:]
# dff2=dff.dropna()
# 
# dff2=dff2.reset_index(drop=True)
# 
# # 到這之前 可能會有個錯誤而無法進行 .corr 
# # 'float' object has no attribute 'shape'
# res2.info()
# # 前面有處理  所以沒問題
# # dff2=dff2.astype('float32')  
# 
# 
# import seaborn as sns
# dfData = res2.corr(method ='pearson')
# plt.subplots(figsize=(10, 10)) # 设置画面大小
# sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
# # plt.savefig('./BluesStateRelation.png')
# plt.show()
# 
# # =============================================================================
# # # 熱圖的備案
# # dfcorr=dff2.corr(method ='pearson')
# # plt.pcolor(dfcorr)
# # plt.show()
# # =============================================================================
# =============================================================================
