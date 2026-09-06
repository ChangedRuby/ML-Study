import numpy as np
import matplotlib.pyplot as plt

plt.figure(0)
x = np.linspace(-10,10,100)
for i in range(10):
    y = np.random.randn()*x
    plt.plot(x, y)
plt.grid(True)
plt.figure(1)
for i in range(10):
    y = np.random.randn()*x+np.random.randn()
    plt.plot(x, y)
plt.grid(True)
plt.show()