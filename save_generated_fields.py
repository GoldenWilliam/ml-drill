import matplotlib.pyplot as plt
import numpy as np
from generate_2Dsim import generate_2D_field

with open("training_data.txt",'w') as file:
    for _ in range(10):
        field = generate_2D_field()
        field_string = '\n'.join([''.join(map(str, row)) for row in field])
        print(field_string)

        file.write(field_string)
        file.write("\n\n")

file.close()



