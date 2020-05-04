#%%
import os
# import requests
import pandas as pd
import sqlite3
import time

#%%

# 確認檔案 和 路徑
print(os.getcwd())
csv_list=os.listdir('./csv')

data_list=[]
i = csv_list[0]
for i in csv_list:
    print(i)
    tablename=i.replace('.','')
    
    path = './csv' + '/' + i
    df = pd.read_csv(path) 
    # data_list.append(data)
    
    time.sleep(1)
    
    conn= sqlite3.connect(('./db/csv.db'))
    df.to_sql(tablename, conn, if_exists='replace', index=False)
    
    time.sleep(1)
    conn.close()
    time.sleep(1)
    



    
    

