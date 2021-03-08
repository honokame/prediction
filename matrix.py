#%%
import useful_graphs # 平均用の混合行列用 
import matplotlib.pyplot as plt
import numpy as np
#%%
# RNNの混合行列
plt.figure(dpi=700)
matrix = np.array([
                     [34,6,0,0,0,0],
                     [6,29,5,0,0,0],
                     [2,15,20,3,0,0],
                     [0,0,4,24,9,3],
                     [0,2,6,17,10,5],
                     [0,0,1,3,6,30]
                   ])
classes = ['100g','200g','300g','500g','600g','700g']
cm = useful_graphs.ConfusionMatrix(matrix, class_list=classes)
cm.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='/home/honoka/research/prediction/result/rnn/matrix_rnn2.png')
#%%
# LSTMの混合行列
plt.figure(dpi=700)
matrix = np.array([
                     [35,4,1,0,0,0],
                     [6,28,6,0,0,0],
                     [2,4,27,5,2,0],
                     [0,0,3,27,9,1],
                     [0,0,4,13,18,5],
                     [0,0,1,0,6,33]
                   ])
classes = ['100g','200g','300g','500g','600g','700g']
cm = useful_graphs.ConfusionMatrix(matrix,class_list=classes)
cm.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='/home/honoka/research/prediction/result/lstm/matrix_lstm2.png')
#%%
# Self-Attentionの混合行列
plt.figure(dpi=700)
matrix = np.array([
                     [37,2,1,0,0,0],
                     [6,27,7,0,0,0],
                     [3,5,26,3,3,0],
                     [0,0,3,24,11,2],
                     [0,1,6,12,16,5],
                     [0,0,1,1,5,33]
                   ])
classes = ['100g','200g','300g','500g','600g','700g']
cm = useful_graphs.ConfusionMatrix(matrix,class_list=classes)
cm.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='/home/honoka/research/prediction/result/self-attention/matrix_self-attention2.png')
#%%
# 分解能の混合行列
plt.figure(dpi=700)
matrix = np.array([
                     [37,3,0],
                     [4,32,4],
                     [1,3,36]
                   ])
classes = ['300g','500g','700g']
cm = useful_graphs.ConfusionMatrix(matrix,class_list=classes)
cm.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='/home/honoka/research/prediction/result/p4/matrix_p4.png')