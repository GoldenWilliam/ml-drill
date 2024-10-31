import numpy as np
import random

class GetFields():
    """Save traningdata read from a txt file in one list. Using get_field we can get one data set"""

    def __init__(self,train=True):
        # For saving the datasets
        self.training_data = []

        # To get out next data set
        self.i = -1

        # For random shuffeling training data
        self.train = train

    def load_data(self, filename: str):
        for block in open(filename).read().split("\n\n"):
            field = []
            for row in block.splitlines():
                array = [int(num) for num in row if num.strip()]
                if len(array) != 0:
                    field.append(array)

            if len(field) != 0: 
                self.training_data.append(field)
        
    def get_field(self):
        self.i += 1
        # if not self.train:
            # print(f"i : {self.i}")
        if self.i >= len(self.training_data) or self.i == 0:
            # Random shuffle the list and start for the beginning
            self.i = 0
            if self.train:
                random.shuffle(self.training_data)

        return np.asarray(self.training_data[self.i])
        



