import numpy as np
import matplotlib.pyplot as plt


def leakyrelu(x):      # Leaky ReLU(Rectified Linear Unit)
    return np.maximum(0.1 * x, x)  #same


x = np.arange(-5, 5, 0.1)
y = leakyrelu(x)

plt.plot(x, leakyrelu(x), linestyle = '-', label = 'Leaky ReLU')
plt.grid()
plt.show()