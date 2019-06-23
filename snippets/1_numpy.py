import numpy as np
import pandas as pd

ndarr = pd.DataFrame(data=np.ndarray(shape=(200,2), dtype=float, order='F'), columns=['first','second'])
rand_ix = np.random.choice(ndarr.index, size=int(len(ndarr)*0.9), replace=False)
print(ndarr.iloc[2])

print(ndarr.iloc[rand_ix[2]])