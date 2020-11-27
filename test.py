#%%
import numpy as np

#%%
def softmax(a):
  c = np.max(a)
  exp = np.exp(a - c)
  sum_exp = np.sum(np.exp(a - c))
  y = exp / sum_exp

  return y,sum(y)

#%%
a = np.array([1010,1000,990])
print(softmax(a))
b = np.array([0.3,2.9,4.0])
print(softmax(b))
# %%
csv = np.