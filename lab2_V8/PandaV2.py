import pandas as pd
from datetime import datetime, timedelta

n = 3

orders = pd.read_csv('C:/Users/szabila/Desktop/PaytonMayp/lab2_V8/orders.csv')
customers = pd.read_csv('C:/Users/szabila/Desktop/PaytonMayp/lab2_V8/customers.csv')

orders['Order Date'] = pd.to_datetime(orders['Order Date'])

last_n_years = datetime.now() - timedelta(days=365 * n)
recent_orders = orders[orders['Order Date'] >= last_n_years]

first_class_orders = recent_orders[recent_orders['Ship Mode'] == 'First Class']
num_first_class_orders = len(first_class_orders)

california_customers = customers[customers['State'] == 'California']
num_california_customers = len(california_customers)

california_orders = recent_orders[recent_orders['Customer ID'].isin(california_customers['Customer ID'])]
num_california_orders = len(california_orders)

orders = orders.merge(customers[['Customer ID', 'State']], on='Customer ID')
recent_orders = orders[orders['Order Date'] >= last_n_years]
pivot_table = recent_orders.groupby('State')['Sales'].mean().reset_index()
pivot_table.columns = ['State', 'Average Check']

print("Количество заказов, отправленных первым классом за последние", n, "лет:", num_first_class_orders)
print("Количество клиентов из Калифорнии в базе данных:", num_california_customers)
print("Количество заказов, сделанных клиентами из Калифорнии за последние", n, "лет:", num_california_orders)
print("\nСводная таблица средних чеков по штатам за последние", n, "лет:")
print(pivot_table)
