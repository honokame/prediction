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
a = [[1, 2], [3, 4], [5, 6]]
b = [1, 2, 3, 4, 5, 6]
print(a)
print(b)

# %%