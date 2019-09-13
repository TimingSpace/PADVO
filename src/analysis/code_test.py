import data.transformation as tf
import numpy as np

a = 0.12 * np.ones((20,6))
b = tf.se_mean(a)
print(b)

