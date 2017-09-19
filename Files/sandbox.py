import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])

def plus_one(x):
	return x*5

print df
print(df[:, None])

'''
1) sigmoid
2) forward pass in the train method
3) backpropin train methos
4) forward pass in run method
'''