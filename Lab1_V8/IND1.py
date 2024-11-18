import numpy as np

array = np.array([
    [1, 2+3j, 3],
    [4.5, 5, 6+0j],
    [7, 8, 9]
])

is_complex = np.iscomplex(array)

is_real = np.isreal(array)

is_scalar = np.vectorize(np.isscalar)(array)

print("Исходный массив:")
print(array)

print("\nЯвляется ли элемент комплексным:")
print(is_complex)

print("\nЯвляется ли элемент действительным:")
print(is_real)

print("\nЯвляется ли элемент скаляром:")
print(is_scalar)
