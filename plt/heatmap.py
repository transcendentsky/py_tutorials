import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# y = np.random.randint(1,100,40)
# y = y.reshape((5,8))
# df = pd.DataFrame(y, columns=[x for x in 'abcdefg'])
# sns.heatmap(df, annot=True)
# plt.show()

np.random.seed(20180316)
x = np.random.randn(4,4)
f, (ax1) = plt.subplots(figsize=(6,6), nrows=1)
sns.heatmap(x, annot=True, ax=ax1)

plt.show()