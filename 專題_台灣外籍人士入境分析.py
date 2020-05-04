#加入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#修改後可正確顯示圖表內的中文字
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 

#讀取檔案
data = r"C:/Terry/歷年來台旅客國籍統計_test.csv"
#指定","為千分位符號即可解除問題
#pandas出現"error tokenizing data", 加上error_bad_lines=False跳過錯誤資料
df = pd.read_csv(data, encoding = "ANSI",  thousands=",")
#df["國籍"]

"""轉置矩陣
df2 = df.stack()
print(df2)
df3 = df2.unstack(0)
print(df3)
"""

#df.set_index(inplace = True) #更換index

#建立基礎圖表(亞洲)
"""將資料鎖定在"亞洲地區", "2002年" 並以人數排序取前十名, 
sort_values =>降冪排序, tail(10)取前10, 移除不要的資料(亞洲合計)"""

#for i in range(2002, 2016):
    #asia = df[df["國籍"]=="亞洲地區"].sort_values(str(i)).iloc[:-1,:]

    
    
    
asia = df[df["國籍"]=="亞洲地區"].sort_values("2002").iloc[:-1,:]

#設定畫布大小
fig, ax = plt.subplots(figsize=(15, 8)) 

#設定對應顏色
colors = dict(zip(
    ["亞洲其他地區 Others", "中東 Middle East", "印度 India", "東南亞地區 東南亞其他地區 Others", "東南亞地區 馬來西亞 Malaysia", "東南亞地區 菲律賓 Philippines", "東南亞地區 新加坡 Singapore","韓國 Korea,Republic of","東南亞地區 印尼 Indonesia","東南亞地區 越南 Vietnam","東南亞地區 泰國 Thailand","東南亞地區 東南亞小計 Sub-Total","日本 Japan"],
    ["#2E86AB", "#424B54", "#00A6A6", "#F24236", "#9E643C", "#f7bb5f", "#EDE6F2", "#E9D985", "#8C4843", "#90d595", "#e48381", "#090446", "#f7bb5f"])) 
#print("加入顏色成功")
#橫向長條圖
ax.barh(asia["細分"], asia["2002"], 
        color=[colors[x] for x in asia["細分"]],height=0.8)

dx = asia["2002"].max() / 200

for i, (value, name) in enumerate(zip(asia["2002"], asia["細分"])):
    ax.text(0, i,name+' ',size=16, weight=600, ha='right', va='center') #將每個國家名稱放到對應的長條旁邊
    ax.text(value+dx, i,f'{value:,.0f}',  size=16, ha='left',  va='center')


ax.text(0, 1.06, "來台人數 (thousands)", transform=ax.transAxes, size=14, color='#777777') #小標題 
ax.text(0, 1.15, "歷年來台旅客國籍統計",transform=ax.transAxes, size=24, weight=600, ha='left', va='top') #標題
ax.text(0.59, 0.14, '總來台人數:'+str(int(asia['2002'].sum())), transform=ax.transAxes, size=30, color='#000000',ha='left') #計算總來台人數
ax.grid(which='major', axis='x', linestyle='-') #增加格線
ax.tick_params(axis='x', colors='#777777', labelsize=12) #調整橫軸顏色、標籤大小
ax.set_yticks([])
ax.margins(0, 0.01)
ax.show()



