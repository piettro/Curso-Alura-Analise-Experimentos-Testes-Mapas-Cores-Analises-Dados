import pandas as pd
import numpy as np
import pyDOE2 as doe
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.cm as cm

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

def model_recipe(x_f, x_c, params):
    #limit
    limit_normalize = [-1,1]
    limit_farinha = [0.5,1.5]
    limit_chocolate = [0.1, 0.5]

    #convert
    x_f_converted = np.interp(x_f, limit_farinha, limit_normalize)
    x_c_converted = np.interp(x_c, limit_chocolate, limit_normalize)

    porcoes = params['Intercept'] + params['Farinha']*x_f_converted + params['Chocolate']*x_c_converted

    return round(porcoes)

def generate_model(equation):
    model = smf.ols(data=experiment, formula=equation)
    adjusted_model = model.fit()
    print(adjusted_model.summary())

    distribution_t = stats.t(df=adjusted_model.df_model)
    name = adjusted_model.tvalues.index.tolist()
    limit = [distribution_t.ppf(q=1-0.025)] * len(name)

    pareto = sns.barplot(x=adjusted_model.tvalues, y=name)
    pareto.figure.set_size_inches(15,6)
    pareto.tick_params(labelsize=20)
    pareto.set_xlabel('t-values', fontsize=20)
    pareto.plot(limit, name ,'r')
    plt.show()

    y = experiment['Porcoes']
    predict_y = adjusted_model.predict()

    #Scatter
    plt.figure(figsize=(10,5))
    plt.xlabel('Predict', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    x_guide_line = np.linspace(start=15, stop=50, num=10)
    y_guide_line = np.linspace(start=15, stop=50, num=10)
    plt.plot(x_guide_line,y_guide_line,'r')
    plt.scatter(predict_y, y)
    plt.show()

    params = adjusted_model.params

    print(model_recipe(0.5,0.1,params))

    x_farinha = np.linspace(start=0.5, stop=1.5, num=10)
    x_chocolate = np.linspace(start=0.1, stop=0.5, num=10)

    dots = []

    for cont1 in x_farinha:
        temp = []

        for cont2 in x_chocolate:
            temp.append(model_recipe(cont1,cont2))

        dots.append(temp)

    #Color Map
    plt.figure(figsize=(16,6))
    plt.xlabel('Farinha (kg)', fontsize=16)
    plt.ylabel('Chocolate (kg)', fontsize=16)
    plt.imshow(dots, origin='lower', cmap=cm.rainbow, interpolation='quadric',extent=(0.5,1.5,0.1,0.5))
    plt.colorbar().set_label('Porcoes',fontsize=16)
    lines = plt.contour(x_farinha,x_chocolate,dots,colors='k',linewidhts=1.5)
    plt.clabel(lines,inline=True,fontsize=15,inline_spacing=10,fmt='%1.0f')
    plt.show()


##===============================================================
## Models V1
equation = 'Porcoes ~ Farinha + Chocolate + Farinha:Chocolate'
model = generate_model(equation)

## Models V2
equation = 'Porcoes ~ Farinha + Chocolate'
model = generate_model(equation)
