#%%
import requests
import pandas as pd
import sqlite3
import time
#%%
"""
將股票交易資料放進Mysql資料庫，
就不用需要每次從證交所讀取資料
從網址
http://www.twse.com.tw/exchangeReport/STOCK_DAY?date=20180817&stockNo=2330
讀取股票交易資料，
該網址會以JSON格式回傳股票編號2330指定日期20180817的一個月股票交易資料，
再進行處理放進Mysql資料庫。
"""
urlhead='http://www.twse.com.tw/exchangeReport/STOCK_DAY?date='

yymmll=list(pd.date_range(start='1/1/2010', end='12/1/2019', freq='M') .strftime('%Y%m%d'))

urlmid='&stockNo='
stockNo='2330'

i=yymmll[0]
for i in yymmll:
    time.sleep(5)
    
    strurl = urlhead + i + urlmid + stockNo
    r = requests.get(strurl) # 使用 GET 方式下載普通網頁
    
    # 確認連線狀態
    if r.status_code == requests.codes.ok:
        print(i,"連線: OK")
    
    data = r.json()
    df = pd.DataFrame( data["data"], columns=data["fields"])
    
    
    # 有時候在連城寫code 有時候在家寫code 一直設工作目錄很煩  
    # 直接開專案架構 使用相對路徑
    conn= sqlite3.connect(("./db/stockNo"+stockNo+'.db'))
    df.to_sql(('stockNo'+stockNo+'_'+i), conn, if_exists='replace', index=False)
    print(('stockNo'+stockNo+'_'+i))
    
    # 剛寫好的時候 迴圈跑很快 然後就被証交所網站擋IP了
    time.sleep(2)
    
    conn.close()
    time.sleep(5)
    
    
    
    
    






