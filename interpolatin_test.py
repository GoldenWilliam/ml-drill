import numpy as np
import matplotlib.pyplot  as plt


grid = np.random.randint(0,7,size=(5,5))
print(grid)

a = np.arange(0,256,256/7)

for i in range(7):
    grid[grid == i] = a[i]

print(grid)

print(a)

