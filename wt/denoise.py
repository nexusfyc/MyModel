import pandas as pd
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

df = pd.read_csv('/method/wt/sentiment_data.csv')
data = df['neu']



data_denoise = denoise_wavelet(data, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')

plt.plot(data, color = 'grey',label="Origin", markersize=10)
plt.plot(data_denoise, color = 'orange',label="Denoise", markersize=10)
plt.tick_params(labelsize=16)
plt.legend(loc = "upper right", prop=dict(weight='bold', size=15)) # 图例
plt.savefig("./wt", dpi=300)
plt.show()