import matplotlib.pyplot as plt
import numpy as np

x = 1
x = str(x)
data = np.genfromtxt("progress"+x+'.csv', delimiter=",", names=["x", "y","z"])
plt.plot(data['x'], data['y'])
plt.plot(data['x'], data['z'])
plt.show()
