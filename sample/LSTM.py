#%%
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.initializers import RandomUniform
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model

#%%
csv_double300 = np.loadtxt("/home/honoka/research/prediction/csv/300-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double500 = np.loadtxt("/home/honoka/research/prediction/csv/500-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double700 = np.loadtxt("/home/honoka/research/prediction/csv/700-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
#csvファイルから読み込み
#BOM付きCSVの場合 encoding='utf_8_sig' を指定する必要がある

csv_double300 = np.delete(csv_double300, 0, axis=0)
csv_double500 = np.delete(csv_double500, 0, axis=0)
csv_double700 = np.delete(csv_double700, 0, axis=0)
#一番左の時間の行を削除

#%%
length_partial =30
#実験に使う長さを指定

data = []
target = []
#dataには入力データを targetには対応する荷重を入れる


for i in range(csv_double300.shape[0]):
    data.append(csv_double300[i][0:length_partial])
    target.append(0.3)
    
for i in range(csv_double500.shape[0]):
    data.append(csv_double500[i][0:length_partial])
    target.append(0.5)
    
for i in range(csv_double700.shape[0]):
    data.append(csv_double700[i][0:length_partial])
    target.append(0.7)
    
#300, 500, 700を正規化した0.3, 0.5, 0.7を用いる

#%%
X = np.array(data).reshape(len(data), length_partial, 1)
Y = np.array(target).reshape(len(target), 1)
#学習で使用できる形に変更

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(len(data)*0.2))
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=int(len(X_test)*0.5))
#X, Yそれぞれtrain, test validに分ける(比は8:1:1)

#%%
n_in = len(X[0][0])
#単位時間に入力する大きさ
n_hidden = 40
#中間層の大きさ
n_out = len(Y[0]) 
#出力の大きさ

#%%
model = Sequential()
model.add(LSTM(n_hidden, input_shape=(length_partial, n_in)))
#LSTM層を追加
model.add(Dense(n_out))
#出力層を追加
model.add(Activation('sigmoid'))
#回帰問題なので活性化関数にsigmoidを使用
model.summary()
#モデルの詳細を表示

#%%
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999)
#optimizerとしてAdamを使用
model.compile(loss='mean_squared_error', optimizer=optimizer)
#回帰問題なので損失関数に平均二乗誤差を使用

#%%
epochs = 200
batch_size = 32
#エポック数, batch_sizeを指定
result = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_valid, Y_valid))
#学習を開始する

#%%
#学習結果の可視化
#print(result.history.keys()) → dict_keys(['val_loss', 'loss'])
pp = PdfPages('lstm_loss.pdf')
#pdf出力をする場合は使用

plt.plot(range(1, epochs+1), result.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#グラフ

pp.savefig()
pp.close()
#pdf出力をする場合は使用

#%%
def regress_to_category(regress):
    category_list = [0.3, 0.5, 0.7]
    category = []
    for i in regress:
        temp = np.abs(np.asarray(category_list) - i).argmin()
        category.append(category_list[temp])
        
    return category

#0.3, 0.5, 0.7のうち一番差の絶対値が近い値を返す関数

#%%
regress_train = model.predict(X_train)
#学習結果を用いてX_trainから予測

category_train = regress_to_category(regress_train)
#予測したものをカテゴリ化

regress_test = model.predict(X_test)
#学習結果を用いてX_testから予測

category_test = regress_to_category(regress_test)
#予測したものをカテゴリ化

count_train = 0
count_test = 0
for i in range(len(category_train)):
    if (category_train[i] == Y_train[i]):
        count_train += 1
        
for i in range(len(category_test)):
    if (category_test[i] == Y_test[i]):
        count_test += 1

print("train_acc: ", end="")
print(count_train/len(category_train))
print("test_acc: ", end="")
print(count_test/len(category_test))
#trainとtestの両方について結果を表示
# %%
