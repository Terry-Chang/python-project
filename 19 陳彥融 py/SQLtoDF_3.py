#%%
"""
參考網頁
https://stackoverflow.com/questions/36028759/how-to-open-and-convert-sqlite-database-to-pandas-dataframe

從SQL的表 撈出 做成PD 再試整理
"""

import os
import copy
import time
import sqlite3
import numpy as np
import pandas as pd



#%%
# 確認檔案 和 路徑
print(os.getcwd())
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
    time.sleep(0.15)
    

cnx.close()
time.sleep(0.5)


dfall = pd.concat(Tdatall)
dfall = dfall.reset_index(drop=True)

# 清除不用的變數
del Tdatall, Tnamelist, df, i, cursor, path

#%%
# 改英文 colname
ch = pd.DataFrame(dfall.columns, columns=['ch'])
en = pd.DataFrame(['date', 'shares', 'amount', 'open', 'high', 'low', 'close', 'change', 'turnover'], 
                  columns=['en'])
ch_en = pd.concat([ch,en], axis=1)

del ch, en

dfall.columns = ch_en['en']

#%%
# 確認一下 型態  
# 兩種偷看型態法 很重要
dfall.info()
dfall.dtypes


# 先處理字串裡的 逗號問題
dfall.iloc[:,1:] = dfall.iloc[:,1:].apply(
    lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), 
                            errors='coerce'))

# 再處理 .dtypes 的問題  dtypes => object 有些數學模型是不吃的
dff1 = dfall.iloc[:,0:1]
dff2 = dfall.iloc[:,1:]
dff2=dff2.astype('float32')

dff2.info()
dff2.dtypes


dfall = pd.concat([dff1, dff2], axis=1)
dfall.info()

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
type(df_date.yy)
df_date['ADyear'] = df_date.yy.apply(lambda x: str(int(x)+1911))
# 透過 apply 函式
# 我們把一個匿名函式 lambda 套用到整個 df.Survived Series 之上
# 並以此建立一個新的 存活 欄位。

# 其中 lambda x:  為匿名函式的開頭 也稱 宣告
# 後面接要 return 的東西


df_date.head(5)
df_date['ADate'] = df_date['ADyear'] + '-' + df_date['mm'] + '-' + df_date['dd']
df_date.head(5)

dfall = pd.merge(df_date,dfall, on='date')
dfall.dtypes

dfall.columns
dfall = dfall.iloc[:,5:]

del df_date

#%%
# 接上 CSV.db => 美股台積資料 台灣加權指(大盤) 

print(os.listdir('./csv'))


df_NYSE_TSM = pd.read_csv('./csv/NYSE_TSM.csv')
df_NYSE_TSM['NY_up_down'] = df_NYSE_TSM['Close']-df_NYSE_TSM['Close'].shift()
df_NYSE_TSM.info()

df_TWII = pd.read_csv('./csv/^TWII.csv')
df_TWII['TWII_up_down'] = df_TWII['Close'] - df_TWII['Close'].shift()
df_TWII.info()


# 台積美股 先和 台灣大盤  合併
res1 = pd.merge(df_TWII.loc[:, ['Date','TWII_up_down']], 
               df_NYSE_TSM.loc[:, ['Date','NY_up_down']],
               on=['Date'],
               suffixes=['_TWII','_NYSE'],
               how='left')

del df_TWII, df_NYSE_TSM

#%%
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

#########################
type(dfall)
type(dfall.shape) # 這兩行  行為居然不太一樣
type(dfall['shares']) # 這兩行  行為居然不太一樣
dfall.info()
dfall.dtypes

#########################    


# dfall['gg'] = dfall.shape.apply(lambda x: int(x/1000)) 
# 上面這行 報錯  說是tuple 不能動  
# 下面這行可以用
dfall['vol'] = dfall['shares'].apply(lambda x: int(x/1000))


dfall2 = dfall.iloc[:, [0, -4, -3, -1]]

# 改col名
coll=list(dfall2.columns)
coll[0]="Date"
dfall2.columns=coll

# dfall2 = dfall2.rename(columns={"Date": "ADdate"}) 官網教的爛東西 特難用
res2 = pd.merge(dfall2, res1, on='Date', how='left')
# shift函數是對數據進行移動的操作


# 開始平移美股
res2.columns
res2['NY_up_down'] = res2['NY_up_down'].shift()

del coll

#%%
# KD 指標
"""
未成熟隨機值(RSV)：
(今日收盤價 - 最近9天的最低價) / (最近9天最高價 - 最近9天的最低價)

當日K值：前日K值 * (2/3) + 當日RSV值 * (1/3)
當日D值：前日D值 * (2/3) + 當日K值 * (1/3)
"""

import talib

kd = pd.DataFrame()
kd['k'], kd['d'] = talib.STOCH(dfall['high'], dfall['low'], dfall['close'])

res2 = pd.concat([res2,kd], axis=1)
"""
KD 值可客觀的表現 市場過熱或過冷
最常使用的 2 種時機是：

1.KD 大於 80 時，為高檔超買訊號，市場過熱，股價要開始跌了。
2.KD 小於 20 時，為低檔超賣訊號，市場過冷，股價要開始漲了。
"""
import matplotlib.pyplot as plt

kd_df = res2.query("k > 80 & d > 80")

plt.xlabel('change')
plt.ylabel('Probability')
plt.title('(if k>80 & d>80) density')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
plt.grid(True)
plt.hist(kd_df['change'], 50, density=True, facecolor='g', alpha=0.75)
plt.show()
# 看起來 跌也多 漲也多  關連不大

#======================================
kd_df = res2.query("k < 20 & d < 20")

plt.xlabel('change')
plt.ylabel('Probability')
plt.title('(if k<20 & d<20) density')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
plt.grid(True)
plt.hist(kd_df['change'], 50, density=True, facecolor='g', alpha=0.75)
plt.show()

# 跟股票明師講得都不一樣

del kd, kd_df

#%%
# 線性迴歸
import seaborn as sns

lm_df = copy.deepcopy(res2)
lm_df = lm_df.dropna()
lm_df['k']=lm_df['k'].apply(lambda x: int(x/10))


gridobj = sns.lmplot(x="k", y="change", data=lm_df, 
                     height=7, aspect=1.6, robust=True, palette='tab10', 
                     scatter_kws=dict(s=60, linewidths=1, edgecolors='black'))

# gridobj.set(xlim=(0.5, 7.5), ylim=(0, 10))
plt.title("regression line (k/10) (up_down)", fontsize=20)
plt.show()

del lm_df, gridobj


# =============================================================================

lm_df = copy.deepcopy(res2)
lm_df = lm_df.dropna()
lm_df['d']=lm_df['d'].apply(lambda x: int(x/10))


gridobj = sns.lmplot(x="d", y="change", data=lm_df, 
                     height=7, aspect=1.6, robust=True, palette='tab10', 
                     scatter_kws=dict(s=60, linewidths=1, edgecolors='black'))

# gridobj.set(xlim=(0.5, 7.5), ylim=(0, 10))
plt.title("regression line (d/10) (up_down)", fontsize=20)
plt.show()

del lm_df, gridobj




#%%
# KD密度總圖
kd_df = copy.deepcopy(res2) 


plt.figure(figsize=(16,10), dpi= 80)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 1, "change"], shade=True, label="k=1X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 2, "change"], shade=True, label="k=2X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 3, "change"], shade=True, label="k=3X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 4, "change"], shade=True, label="k=4X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 5, "change"], shade=True, label="k=5X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 6, "change"], shade=True, label="k=6X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 8, "change"], shade=True, label="k=8X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['k']//10 == 9, "change"], shade=True, label="k=9X", alpha=.7)

plt.title('UP DOWN for K (Probability Density Function)', fontsize=22)
plt.legend()
plt.show()

##

plt.figure(figsize=(16,10), dpi= 80)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 1, "change"], shade=True, label="d=1X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 2, "change"], shade=True, label="d=2X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 3, "change"], shade=True, label="d=3X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 4, "change"], shade=True, label="d=4X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 5, "change"], shade=True, label="d=5X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 6, "change"], shade=True, label="d=6X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 8, "change"], shade=True, label="d=8X", alpha=.7)
sns.kdeplot(kd_df.loc[kd_df['d']//10 == 9, "change"], shade=True, label="d=9X", alpha=.7)

plt.title('UP DOWN for D (Probability Density Function)', fontsize=22)
plt.legend()
plt.show()


#%%
# 做個三維圖

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 畫子圖相關


df3D = copy.deepcopy(res2)
df3D = df3D.dropna()

xs = df3D['k']
ys = df3D['d']
zs = df3D['change']
ax.scatter(xs, ys, zs, marker='.')


ax.set_xlabel('k')
ax.set_ylabel('d')
ax.set_zlabel('up down')

plt.show()



#%%
# 資料還沒撈完  先做看看熱圖
"""
相關係數
在生物醫學的研究中，常需對感興趣的兩個變數來檢驗是否具有相關性
以及如具有相關性，其相關的方向爲正向或是反向相關？
例如：體脂肪與血壓，體重與血糖值高低之間的關聯性。
而最常被應用來呈現相關性的指標即爲pearson 相關或spearman相關，
然而這兩個指標在應用的情形上有所不同。

一般而言，Pearson 相關常用來呈現連續型(continous)變數之間的關聯性
尤其在變數符合常態分配的假設下，最爲精確；

而spearman相關則不需符合常態，僅要求變數的資料型態至少爲有序的(ordinal)。
另一個選擇上的重點爲在資料具有離群值時(outliers)，
以spearman相關來呈現會是較佳的選擇，
因爲其不受離群值的影響(這是因爲spearman相關是以排序值(rank)來計算相關係數！)

更深入的來看，pearson相關所衡量的是”線性”相關(linear)，
也就是說，主要偵測的是兩變數之間是否有線性相關。

所以，當兩變數之間具有相關，但爲非線性時pearson就不是最佳的方法
在這種情形下，spearman更爲合適


DataFrame.corr(self, method='pearson', min_periods=1) 
Parameters
method{‘pearson’, ‘kendall’, ‘spearman’} or callable
"""
# 定義一個函數 做corr 然後畫圖
def hotmapp (dff,method_str):
    print(res2.info())
    print(res2.dtypes)
    
    import matplotlib.pyplot as plt
    
    corr_arr = dff.corr(method = method_str)
    
    plt.subplots(figsize=(12, 12)) # 设置画面大小
    
    import seaborn as sns
    sns.heatmap(corr_arr, annot=True, vmax=1, square=True, cmap="Blues")
    plt.yticks(rotation=0) 
    
    return plt.show()
    

hotmapp(res2, 'spearman')



#%%
# 畫k線

import pandas_datareader as pdr
# visual
import matplotlib.pyplot as plt
import mpl_finance as mpf
# %matplotlib inline
import seaborn as sns

#time
import datetime as datetime

#talib
import talib



df_2330 = copy.deepcopy(dfall.iloc[-365:,:])
df_2330.info()

# df_2330.index = df_2330['Date'].format(formatter=lambda x: x.strftime('%Y-%m-%d')) 

fig = plt.figure(figsize=(24, 8))

ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(0, len(df_2330['ADate']), 10))
plt.grid(True)
plt.xticks(rotation=70)
# ax.set_xticklabels(df_2330.index[::10])
ax.set_xticklabels(df_2330['ADate'])
mpf.candlestick2_ochl(ax, df_2330['open'], df_2330['close'], df_2330['high'],
                      df_2330['low'], width=0.6, colorup='r', colordown='g', alpha=0.75); 


#%%



