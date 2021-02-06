#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from keras.utils import np_utils
from sklearn.model_selection import train_test_split  #データセットの分割
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.core import Activation
from keras.initializers import glorot_normal
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.optimizers import SGD
from matplotlib.backends.backend_pdf import PdfPages
from keras.utils import plot_model


#%%
#csvファイル読み込み
#BOM付きなのでencoding="utf_8_sig"を指定
csv100 = np.loadtxt("/home/honoka/research/prediction/csv/500_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv200 = np.loadtxt("/home/honoka/research/prediction/csv/600_1.csv", delimiter=",", encoding='utf_8_sig',unpack=True)
csv300 = np.loadtxt("/home/honoka/research/prediction/csv/700_1.csv", delimiter=",", encoding='utf_8_sig', unpack=True)

#時間の行を削除
csv100 = np.delete(csv100, 0, 0)
csv200 = np.delete(csv200, 0, 0)
csv300 = np.delete(csv300, 0, 0)

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

# %%
#学習できる形に変換
x = np.array(data)
x = np.array(data).reshape(len(data),length,1)
t = np.array(target).reshape(len(target), 1)
t = np_utils.to_categorical(t)

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=int(len(data) * 0.2),stratify=t)
x_valid, x_test, t_valid, t_test = train_test_split(x_test, t_test, test_size=int(len(x_test) * 0.5),stratify=t_test)
#x_train2, x_test2, t_train2, t_test2 = train_test_split(x[200:399], t[200:399], test_size=int(200 * 0.2))
#x_valid2, x_test2, t_valid2, t_test2 = train_test_split(x_test2[200:399], t_test2[200:399], test_size=int(len(x_test2)) * 0.5))


# %%
#入力、隠れ、出力のユニット数
l_in = len(x[0])  #301
l_hidden = 10
l_out = 3
# %%
#モデルの構築
model = Sequential()  #入力と出力が１つずつ
model.add(SimpleRNN(l_hidden,input_shape=(length,1)))#隠れ層のユニット数、活性化関数、入力の形
model.add(Dense(l_out, activation='softmax')) #多クラス分類なのでソフトマックス関数、シグモイドも試す？
model.summary()

#model1 = Sequential()
#model1.add(SimpleRNN(l_hidden, input_shape=(91, 10)))
#model1.add(Dense(l_out, activation='softmax'))
#model1.summary()
# %%
#学習の最適化
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)  #後日パラメータ調整
#損失関数（交差エントロピー誤差）、最適化アルゴリズム、評価関数
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 
#model1.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
#%%
#学習開始
batch_size = 32
epochs = 100
result = model.fit(x_train,t_train, batch_size=batch_size,epochs=epochs,validation_data=(x_valid, t_valid))
#result1 = model1.fit(x_train,t_train,batch_size=16,epochs=epochs,validation_data=(x_valid,t_valid))
#%%
#過学習チェック
pp = PdfPages('rnn_accuracy.pdf')
plt.plot(range(1, epochs+1), result.history['accuracy'], label="train_acc")
plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="valid_acc")
#plt.plot(range(1,epochs+1),result1.history['accuracy'],label="normal")
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
pp.savefig()
pp.close()


# %%
#学習結果の可視化
pp = PdfPages('rnn_loss.pdf')
plt.plot(range(1, epochs+1), result.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation_loss")
#plt.plot(range(1,epochs+1),result1.history['loss'],label="normal")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
pp.savefig()
pp.close()

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

print(epochs,l_hidden,batch_size)
print("train_acc=")
print(count_train / len(score_train))
print("test_acc=")
print(count_test / len(score_test))


# %%
print(len(t_test))
t_test_change = []
for i in range(60):
  t_test_change.append(np.argmax(t_test[i]))

predit_classes = model.predict_classes(x_test)
true_classes = t_test_change

#print(confusion_matrix(true_classes,predit_classes))

def print_mtrix(t_true, t_predict):
  labels = sorted(list(set(t_true)))
  mtrix_data = confusion_matrix(t_true, t_predict, labels=labels)
  
  df_mtrix = pd.DataFrame(mtrix_data, index=labels, columns=labels)
  
  plt.figure(figsize=(12, 7))
  sb.heatmap(df_mtrix, annot=True, fmt='g', square=True,cmap='Blues')
  plt.show()

print_mtrix(true_classes,predit_classes)
# %%
