import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ex2data1.txt", header=None)

df.head()
df.describe()

pos, neg = (y == 1).reshape(100, 1), (y == 0).reshape(100, 1)
