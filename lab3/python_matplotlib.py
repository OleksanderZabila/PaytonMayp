import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла (путь к вашему файлу)
file_path = 'C:\\Users\\szabila\\Desktop\\PaytonMayp\\lab3\\affairs.csv'  # Убедитесь, что путь к файлу корректен
data = pd.read_csv(file_path)

# Гистограмма распределения возраста
plt.figure(figsize=(8, 5))
plt.hist(data['age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Распределение возраста', fontsize=14)
plt.xlabel('Возраст', fontsize=12)
plt.ylabel('Количество людей', fontsize=12)
plt.grid(True)
plt.show()
