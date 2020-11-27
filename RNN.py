#%%
import numpy as np

#%%
#csvファイル読み込み
#BOM付きなのでencoding="utf_8_sig"を指定
csv300 = np.loadtxt("/home/honoka/research/prediction/csv/300.csv", delimiter=",", encoding='utf_8_sig')
print(csv300)
csv500 = np.loadtxt("/home/honoka/research/prediction/csv/500.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
print(csv500)

# %%

