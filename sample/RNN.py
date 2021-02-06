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
csv_double300 = np.loadtxt("/home/honoka/research/prediction/csv_old/300.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double500 = np.loadtxt("/home/honoka/research/prediction/csv_old/500.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')
csv_double700 = np.loadtxt("/home/honoka/research/prediction/csv_old/700.csv", dtype="float", delimiter=",", unpack=True, encoding='utf_8_sig')

#BOM付きCSVの場合 encoding='utf_8_sig' を指定する必要がある

csv_double300 = np.delete(csv_double300, 0, axis=0)
csv_double500 = np.delete(csv_double500, 0, axis=0)
csv_double700 = np.delete(csv_double700, 0, axis=0)
#%%
length_partial =30

data = []
target = []


for i in range(csv_double300.shape[0]):
    data.append(csv_double300[i][0:length_partial])
    target.append(0.3)
    
for i in range(csv_double500.shape[0]):
    data.append(csv_double500[i][0:length_partial])
    target.append(0.5)
    
for i in range(csv_double700.shape[0]):
    data.append(csv_double700[i][0:length_partial])
    target.append(0.7)

#%%
X = np.array(data).reshape(len(data), length_partial, 1)
Y = np.array(target).reshape(len(target), 1)
print(data)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(len(data)*0.2))
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=int(len(X_test) * 0.5))

#%%
n_in = len(X[0][0])
n_hidden = 40
n_out = len(Y[0])
print(len(X[0][0]))
print(len(Y[0]))

#%%
print(length_partial)
print(n_in)
model = Sequential()
model.add(SimpleRNN(n_hidden, input_shape=(length_partial, n_in)))
model.add(Dense(n_out))
model.add(Activation('sigmoid'))
model.summary()

#%%
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)

#%%
epochs = 100
batch_size = 32
result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))

#%%
#学習結果の可視化
#print(result.history.keys()) → dict_keys(['val_loss', 'loss'])
pp = PdfPages('rnn_loss.pdf')
plt.plot(range(1, epochs+1), result.history['loss'], label="training_loss")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
pp.savefig()
pp.close()

#%%
def regress_to_category(regress):
    category_list = [0.3, 0.5, 0.7]
    category = []
    for i in regress:
        temp = np.abs(np.asarray(category_list) - i).argmin()
        category.append(category_list[temp])
        
    return category

#%%
regress_train = model.predict(X_train)
category_train = regress_to_category(regress_train)
regress_test = model.predict(X_test)
category_test = regress_to_category(regress_test)

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