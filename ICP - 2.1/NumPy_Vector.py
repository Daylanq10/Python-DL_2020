import numpy as np

# RANDOM MATRIX IN RANGE 1-20 OF SIZE 20
x = np.random.randint(low=1, high=20, size=20, dtype='int')
print(x)
print()

# RESHAPE PREVIOUS MATRIX TO (4,5)
y = x.reshape((4, 5))
print(y)
print()

# TAKES MAX OF EACH ROW AND MULTIPLIES 0 BY HIGHEST NUMBER
z = np.where(y == np.max(y, axis=1, keepdims=True), 0 * y, y)
print(z)
