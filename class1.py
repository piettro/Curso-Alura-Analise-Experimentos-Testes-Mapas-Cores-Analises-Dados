import pandas as pd
import numpy as np
import pyDOE2 as doe
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

sns.set_palette('terrain')
sns.set_style('darkgrid')

sample = doe.ff2n(2)
experiment = pd.DataFrame(sample, columns=['Farinha','Chocolate'])
experiment['Porcoes'] = [19,37,24,49]

center = np.array([
    [0,0,29],
    [0,0,30],
    [0,0,29],
    [0,0,30],
])

center_dataframe = pd.DataFrame(center, columns=['Farinha','Chocolate','Porcoes'], index=[4,5,6,7])
experiment = experiment._append(center_dataframe)
print(experiment.head(10))

ax1 = sns.lmplot(data=experiment, x='Farinha',y='Porcoes', ci=None, hue='Chocolate')
ax1.set(xticks=(-1,1))
plt.show()

ax2 = sns.lmplot(data=experiment, x='Chocolate',y='Porcoes', ci=None, hue='Farinha')
ax2.set(xticks=(-1,1))
plt.show()

model = smf.ols(data=experiment, formula='Porcoes ~ Farinha + Chocolate + Farinha:Chocolate')
adjusted_model = model.fit()
print(adjusted_model.summary())

distribution_t = stats.t(df=4)
name = adjusted_model.tvalues.index.tolist()
limit = [distribution_t.ppf(q=1-0.025)] * len(name)

pareto = sns.barplot(x=adjusted_model.tvalues, y=name)
pareto.figure.set_size_inches(15,6)
pareto.tick_params(labelsize=20)
pareto.set_xlabel('t-values', fontsize=20)
pareto.plot(limit, name ,'r')
plt.show()