#%%
import numpy as np
from sklearn.model_selection import train_test_split #データセットの分割

#%%
#csvファイル読み込み
#BOM付きなのでencoding="utf_8_sig"を指定
csv300 = np.loadtxt("/home/honoka/research/prediction/csv/300.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv500 = np.loadtxt("/home/honoka/research/prediction/csv/500.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv700 = np.loadtxt("/home/honoka/research/prediction/csv/700.csv", delimiter=",", encoding='utf_8_sig', unpack=True)

#時間の行を削除
csv300 = np.delete(csv300, 0, 0)
csv500 = np.delete(csv500, 0, 0)
csv700 = np.delete(csv700, 0, 0)

# %%
#データを格納、学習に使う長さを指定
length = 30

data = [] #入力値
target = []  #出力値

#入力値と目標値を格納
for i in range(csv300.shape[0]):  #データの数
  data.append(csv300[i][0:length])
  target.append(0.3)

for i in range(csv500.shape[0]):  
  data.append(csv500[i][0:length])
  target.append(0.5)

for i in range(csv700.shape[0]):
  data.append(csv700[i][0:length])
  target.append(0.7)

# %%
#学習できる形に変換
x = np.array(data)
t = np.array(target)
#x = np.array(data).reshape(len(data), length)
#t = np.array(target).reshape(len(target), 1)

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3)
print(t_test.shape)
# %%
