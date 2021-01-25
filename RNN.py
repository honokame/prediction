#%%
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from sklearn.model_selection import train_test_split #データセットの分割
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.core import Activation
from keras.optimizers import Adam
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
length_rnn = 10
sample = length - length_rnn
#x = np.zeros((sample, length_rnn))
x = []
y = []
#print(data)

for i in range(0,len(data)):
  for j in range(0,sample):
    x.append(data[i][j: j + length_rnn])

x = np.array(x).reshape(600,sample, length_rnn)
t = np.array(target).reshape(len(target), 1)
t = np_utils.to_categorical(t)

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=int(len(data) * 0.2))
x_valid, x_test, t_valid, t_test = train_test_split(x_test, t_test, test_size=int(len(x_test) * 0.5))
# %%
#入力、隠れ、出力のユニット数
l_in = len(x[0])  #301
l_hidden = 400
l_out = 3
# %%
#モデルの構築
model = Sequential()  #入力と出力が１つずつ
model.add(SimpleRNN(l_hidden,input_shape=(91,10))) #隠れ層のユニット数、活性化関数、入力の形
#model.add(Flatten())
model.add(Dense(l_out, activation='softmax')) #多クラス分類なのでソフトマックス関数、シグモイドも試す？
#model.add(Flatten())
model.summary()

# %%
#学習の最適化
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)  #後日パラメータ調整
#損失関数（交差エントロピー誤差）、最適化アルゴリズム、評価関数
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 

#%%
#学習開始
batch_size = 16
epochs = 100
print(x_train.shape)
result = model.fit(x_train,t_train, batch_size=batch_size,epochs=epochs,validation_data=(x_valid, t_valid))

#%%
#過学習チェック
pp = PdfPages('rnn_accuracy.pdf')
plt.plot(range(1, epochs+1), result.history['accuracy'], label="train_acc")
plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="valid_acc")
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
