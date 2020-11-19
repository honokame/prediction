#%%
#使用するライブラリのインポート
import numpy as np #数値計算
import matplotlib.pyplot as plt #グラフの描画

from matplotlib.backends.backend_pdf import PdfPages #グラフをPDFに保存
from sklearn.model_selection import train_test_split #機械学習ライブラリ

#ニューラルネットワークライブラリ
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
#csvファイル読み込み
#BOM付きCSVの場合 encoding='utf_8_sig' を指定する必要がある
#(ファイル名、データ型、区切り文字、true=転置、文字指定) 
csv_double300 = np.loadtxt("/home/honoka/research/prediction/csv/300-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double500 = np.loadtxt("/home/honoka/research/prediction/csv/500-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double700 = np.loadtxt("/home/honoka/research/prediction/csv/700-2.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')


#一番左の時間の行を削除
csv_double300 = np.delete(csv_double300, 0, axis=0)
csv_double500 = np.delete(csv_double500, 0, axis=0)
csv_double700 = np.delete(csv_double700, 0, axis=0)

#%%
#データを格納
#実験に使う長さを指定
#時間0-0.29sまで
length_partial =30

#dataには入力データを targetには対応する荷重を入れる
data = [] #入力値　
target = [] #目標値

#300, 500, 700を正規化した0.3, 0.5, 0.7を用いる
#データの列分まで繰り返す
for i in range(csv_double300.shape[0]): 
    data.append(csv_double300[i][0:length_partial])
    target.append(0.3)
    
for i in range(csv_double500.shape[0]):
    data.append(csv_double500[i][0:length_partial])
    target.append(0.5)
    
for i in range(csv_double700.shape[0]):
    data.append(csv_double700[i][0:length_partial])
    target.append(0.7)
#荷重に関係なく全てdataにいれてる(50*3)

#%%
#学習で使用できる形に変更

X = np.array(data).reshape(len(data), length_partial)
Y = np.array(target).reshape(len(target), 1)

""""
#X
[[0.   0.   0.   ... 0.07 0.08 0.07]
 [0.03 0.03 0.02 ... 0.06 0.07 0.07]
 [0.04 0.   0.04 ... 0.07 0.06 0.05]
 ...
 [0.   0.01 0.   ... 0.18 0.2  0.21]
 [0.01 0.   0.02 ... 0.29 0.28 0.28]
 [0.02 0.01 0.02 ... 0.26 0.26 0.27]]

#Y
[0.3 0.3 ... 0.5 0.7]
↓
[[0.3]
   :
 [0.5]
   :
 [0.7]]
"""""
#X, Yそれぞれtrain, test validに分ける(比は8:1:1) 
#len(data) = 150,len(X_test) = 30
#訓練データとテストデータを8:2に分けてからテストデータと検証データを1:1に分けている
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(len(data)*0.2))
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size = int(len(X_test) * 0.5))

#%%
#単位時間に入力する大きさ
n_in = len(X[0])
#中間層の大きさ
n_hidden = 40
#出力の大きさ
n_out = len(Y[0])

#%%
#NN層を追加?
model = Sequential()
model.add(Dense(n_hidden, input_shape=(length_partial, )))
model.add(Activation('relu'))
#出力層を追加
model.add(Dense(n_out))
#回帰問題なので活性化関数にsigmoidを使用
model.add(Activation('sigmoid'))
#モデルの詳細を表示
model.summary()

#%%
#optimizerとしてAdamを使用
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999)
#回帰問題なので損失関数に平均二乗誤差を使用
model.compile(loss = 'mean_squared_error', optimizer = optimizer)

#%%
#エポック数, batch_sizeを指定
epochs = 70
batch_size = 32
#学習を開始する
result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))

#%%
#学習結果の可視化
#print(result.history.keys()) → dict_keys(['val_loss', 'loss'])
#pdf出力をする場合は使用
pp = PdfPages('fnn_loss.pdf')

#グラフ
plt.plot(range(1, epochs+1), result.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#pdf出力をする場合は使用
pp.savefig()
pp.close()

#%%
#0.3, 0.5, 0.7のうち一番差の絶対値が近い値を返す関数
def regress_to_category(regress):
    category_list = [0.3, 0.5, 0.7]
    category = []
    for i in regress:
        temp = np.abs(np.asarray(category_list) - i).argmin()
        category.append(category_list[temp])
        
    return category

#%%
#学習結果を用いてX_trainから予測
regress_train = model.predict(X_train)

#予測したものをカテゴリ化
category_train = regress_to_category(regress_train)

#学習結果を用いてX_testから予測
regress_test = model.predict(X_test)

#予測したものをカテゴリ化
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
# %%
