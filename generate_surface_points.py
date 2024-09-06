import csv
import numpy as np

Id = range(0, 100)
x = np.random.uniform(1, 100, 100)
y = np.random.uniform(1, 100, 100)
z = np.random.uniform(1, 10, 100)
formations = ['s1', 's2', 's3', 's4']
formation = np.random.choice(formations, 100)

with open('surface_points.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('Id', 'X', 'Y', 'Z', 'Formation'))
    writer.writerows(zip(Id, x, y, z, formation))