import numpy as np

array = np.ones((5, 5))

array[1:-1, 1:-1] = 0

print("Исходный массив с 1 на границе и 0 внутри:")
print(array)
