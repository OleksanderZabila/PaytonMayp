import numpy as np

array1 = np.array([0, 1, 2])
array2 = np.array([2, 10])

combined = np.vstack((array1[:len(array2)], array2[:len(array1)]))

cov_matrix = np.cov(combined)

print("Исходный массив1:")
print(array1)

print("Исходный массив2:")
print(array2)

print("\nКовариационная матрица указанных массивов:")
print(cov_matrix)
