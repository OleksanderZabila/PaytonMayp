import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, KBinsDiscretizer

# Загрузка данных
url = 'https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/download'
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

n_bins = 20
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
columns = df.columns
for i, ax in enumerate(axs.flatten()):
    ax.hist(df[columns[i]], bins=n_bins)
    ax.set_title(columns[i])
plt.tight_layout()
plt.show()

data = df.to_numpy(dtype='float')

scaler = StandardScaler().fit(data[:150, :])
data_scaled = scaler.transform(data)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_scaled[:, i], bins=n_bins)
    ax.set_title(columns[i] + ' (standardized)')
plt.tight_layout()
plt.show()

min_max_scaler = MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

max_abs_scaler = MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)

robust_scaler = RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)

quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

power_transformer = PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data)

kbins_discretizer = KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal', strategy='uniform').fit(data)
data_binned = kbins_discretizer.transform(data)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_binned[:, i], bins=n_bins)
    ax.set_title(columns[i] + ' (binned)')
plt.tight_layout()
plt.show()
