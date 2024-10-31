from CNN_enviroment import SoilEnvirment
from load_field_data import GetFields
import random



data = GetFields(train=False)
data.load_data("data/train_data_1.txt")

env = SoilEnvirment(data=data)

env.reset()

for _ in range(10):
    action = random.randint(0,49)
    a,r, d, t, i = env.step(action)
    print("##")
    print(r)
    print(env._calc_rmse())
    print(env.num_holes)
    print(action)
    print("#####")
    
