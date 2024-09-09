import numpy as np

class GetTrainingFields():
    def __init__(self):
        self.training_data = []

        self.get_data()
    
    def get_data(self, filename: str="training_data.txt"):

        for block in open(filename).read().split("\n\n"):
            field = []

            for row in block.splitlines():
                array = [int(num) for num in row]
                if len(array) != 0:
                    field.append(array)

            if len(field) != 0: 
                self.training_data.append(field)
        
        self.training_data = np.asarray(self.training_data)



