import pandas as pd

data = pd.read_csv('C:/Users/szabila/Desktop/PaytonMayp/lab2_V8/customers.csv')

selected_columns = data.iloc[:, [1, 3]]

selected_columns.to_csv('C:/Users/szabila/Desktop/PaytonMayp/lab2_V8/selected_columns.csv', index=False)
