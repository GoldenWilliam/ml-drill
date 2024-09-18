import matplotlib.pyplot as plt
import numpy as np
from generate_2Dsim import generate_2D_field

with open("train_data_200_6.txt",'w') as file:
    for _ in range(200):
        field = generate_2D_field()
        field_string = '\n'.join([''.join(map(str, row)) for row in field]) 
        
        file.write(field_string)
        file.write("\n\n")

file.close()




