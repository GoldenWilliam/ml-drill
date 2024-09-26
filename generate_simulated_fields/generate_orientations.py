import csv
import numpy as np

Id = range(0, 100)
x = np.random.randint(1, 100, 3)
y = np.random.randint(1, 100, 3)
z = np.random.randint(1, 100, 3)
azimuth = np.random.randint(1, 360, 3)
dip = np.random.randint(0, 360, 3)
polarity = np.random.choice([-1, 1], 3)
formations = ['s1', 's2', 's3', 's4',]
formation = np.random.choice(formations, 3)

with open('../orientations.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('X', 'Y', 'Z', 'Azimuth', 'Dip', 'Polarity', 'Formation'))
    writer.writerows(zip(x, y, z, azimuth, dip, polarity, formation))