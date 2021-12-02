import numpy as np
import scipy.stats as stats

r1 = np.load('results/rewards1.npy').tolist()
r2 = np.load('results/rewards2.npy').tolist()

U1, p = stats.mannwhitneyu(r1, r2)
print(U1, p)



