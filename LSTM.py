#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import useful_graphs

from keras.utils import np_utils
from sklearn.model_selection import train_test_split  #データセットの分割
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.optimizers import Adam
from matplotlib.backends.backend_pdf import PdfPages
from keras.utils import plot_model

#%%
#csvファイル読み込み
#BOM付きなのでencoding="utf_8_sig"を指定
csv100 = np.loadtxt("/home/honoka/research/prediction/csv/100_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv200 = np.loadtxt("/home/honoka/research/prediction/csv/200_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv300 = np.loadtxt("/home/honoka/research/prediction/csv/300_1.csv", delimiter=",", encoding='utf_8_sig', unpack=True)
csv500 = np.loadtxt("/home/honoka/research/prediction/csv/500_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv600 = np.loadtxt("/home/honoka/research/prediction/csv/600_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv700 = np.loadtxt("/home/honoka/research/prediction/csv/700_1.csv", delimiter=",", encoding='utf_8_sig', unpack=True)

#時間の行を削除
csv100 = np.delete(csv100, 0, 0)
csv200 = np.delete(csv200, 0, 0)
csv300 = np.delete(csv300, 0, 0)
csv500 = np.delete(csv500, 0, 0)
csv600 = np.delete(csv600, 0, 0)
csv700 = np.delete(csv700, 0, 0)
# %%
#データを格納、学習に使う長さを指定
length = 101 

data = [] #入力値
target = []  #出力値

#入力値と目標値を格納
for i in range(csv100.shape[0]):  #データの数
  data.append(csv100[i][0:length])
  target.append(0)
for i in range(csv200.shape[0]):  
  data.append(csv200[i][0:length])
  target.append(1)
for i in range(csv300.shape[0]):
  data.append(csv300[i][0:length])
  target.append(2)
for i in range(csv500.shape[0]):  #データの数
  data.append(csv500[i][0:length])
  target.append(3)
for i in range(csv600.shape[0]):  
  data.append(csv600[i][0:length])
  target.append(4)
for i in range(csv700.shape[0]):
  data.append(csv700[i][0:length])
  target.append(5)

# %%
#学習できる形に変換
x = np.array(data)
x = np.array(data).reshape(len(data),length,1)
t = np.array(target).reshape(len(target), 1)
t = np_utils.to_categorical(t)

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=int(len(data) * 0.4),stratify=t)
x_valid, x_test, t_valid, t_test = train_test_split(x_test, t_test, test_size=int(len(x_test) * 0.5),stratify=t_test)
# %%
#入力、隠れ、出力のユニット数
l_in = len(x[0]) 
l_hidden = 20
l_out = 6
# %%
#モデルの構築
model = Sequential()  #入力と出力が１つずつ
model.add(LSTM(l_hidden,input_shape=(l_in,1))) #隠れ層のユニット数、活性化関数、入力の形
model.add(Dense(l_out,activation='softmax')) #多クラス分類なのでソフトマックス関数、シグモイドも試す？
model.summary()

#model1 = Sequential()
#model1.add(LSTM(40, input_shape=(l_in, 1)))
#model1.add(Dense(l_out, activation='softmax'))
#model1.summary()

#model2 = Sequential()
#model2.add(LSTM(50, input_shape=(l_in, 1)))
#model2.add(Dense(l_out, activation='softmax'))
#model2.summary()
# %%
#学習の最適化
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)  #後日パラメータ調整
#損失関数（交差エントロピー誤差）、最適化アルゴリズム、評価関数
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 
#model1.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 
#model2.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 



#%%
#学習開始
batch_size = 16
epochs = 100
result = model.fit(x_train,t_train, batch_size=batch_size,epochs=epochs,validation_data=(x_valid, t_valid),verbose=2)
result1 = model.fit(x_train,t_train, batch_size=32,epochs=epochs,validation_data=(x_valid, t_valid),verbose=2)
result2 = model.fit(x_train,t_train, batch_size=64,epochs=epochs,validation_data=(x_valid, t_valid),verbose=2)



#%%
#過学習チェック
plt.figure(dpi=700)
plt.plot(range(1, epochs+1), result.history['accuracy'], label="train_acc")
plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="valid_acc")
plt.plot(range(1, epochs+1), result1.history['accuracy'], label="train_acc2")
plt.plot(range(1, epochs+1), result1.history['val_accuracy'], label="valid_acc2")
plt.plot(range(1, epochs+1), result2.history['accuracy'], label="train_acc3")
plt.plot(range(1, epochs+1), result2.history['val_accuracy'], label="valid_acc3")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/home/honoka/research/prediction/result/lstm/lstm_accuracy.png')
plt.show()
# %%
#学習結果の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs+1), result.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation_loss")
plt.plot(range(1, epochs+1), result1.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result1.history['val_loss'], label="validation_loss")
#plt.plot(range(1, epochs+1), result2.history['loss'], label="training_loss")
#plt.plot(range(1, epochs+1), result2.history['val_loss'], label="validation_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/honoka/research/prediction/result/lstm/lstm_loss.png')
plt.show()
# %%
#学習モデルを用いてx_trainから予測
score_train = model.predict(x_train)

#学習モデルを用いてx_testから予測
score_test = model.predict(x_test)

#正解率を求める
count_train = 0
count_test = 0

for i in range(len(score_train)):
  if (np.argmax(score_train[i]) == np.argmax(t_train[i])):
    count_train += 1

for i in range(len(score_test)):
  if (np.argmax(score_test[i]) == np.argmax(t_test[i])):
    count_test += 1

print("train_acc=")
print(count_train / len(score_train))
print("test_acc=")
print(count_test / len(score_test))
# %%
def print_mtrix(t_true,t_predict):
  mtrix_data = confusion_matrix(t_true,t_predict)
  df_mtrix = pd.DataFrame(mtrix_data, index=['100g','200g','300g','500g','600g','700g'], columns=['100g','200g','300g','500g','600g','700g'])
  
  plt.figure(dpi=700)
  sb.heatmap(df_mtrix, annot=True, fmt='g', square=True,cmap='Blues')
  plt.title('LSTM')
  plt.xlabel('Predictit label', fontsize=13)
  plt.ylabel('True label', fontsize=13)
  plt.savefig('/home/honoka/research/prediction/result/lstm/lstm.png')
  plt.show()

#%%
t_test_change = []
for i in range(240):
  t_test_change.append(np.argmax(t_test[i]))

predit_classes = model.predict_classes(x_test)
true_classes = t_test_change

print_mtrix(true_classes,predit_classes)

# %%
#matrix = np.array([
#                     [14, 6, 0],
#                     [15, 3, 2],
#                     [8, 2, 10],
#                   ])
#classes = ['100g', '200g', '300g']
#cm = useful_graphs.ConfusionMatrix(matrix, class_list=classes)
#m.plot(to_normalize=True)
