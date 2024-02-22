from sklearn.metrics import mean_squared_error as mse # mse
from sklearn.metrics import mean_absolute_error as mae # mae
from sklearn.metrics import mean_absolute_percentage_error as mape # mape
import numpy as np



def df_to_X_y(df, window_size=14):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)

def plot_predictions1(model, X, y, start=0, end=100):
  predictions = model.predict(X)
  return predictions, mse(y, predictions), mae(y, predictions), mape(y, predictions)