import numpy as np
import matplotlib.pyplot  as plt

x = np.arange(2,0.25,-0.01)
y = -7 + 10**(1/x)

plt.plot(x,y)
plt.ylim(-10,100)
plt.show()