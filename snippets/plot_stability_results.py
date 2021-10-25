"""
simple script for plotting the results of a stability test stroed in a csv file. 
Note that the column names are assumed to be constant as they are set in a previous script 
"""

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np 
import sys
import os


plt.rcParams.update({'font.size': 26})

file_path = '/home/erik/sweden/sonnhammer/GeneSnake/generation/network_generation_algo/results/tables/2021-10-06_stability_analysis_itter10_datasets10_4method_4sizes.csv'

df = pd.read_csv(file_path,sep=",")

df["log lambda"] = np.log10(abs(df["min jtlide_lambda"].values))

fig, ax = plt.subplots(figsize=(16,12))

sns.lineplot(x="size", y="log lambda", markers=True, dashes=False,hue="Method",data=df,\
             hue_order = ["randG", 'gnw', 'nx', "FFL", 'DAG'], ci=95, linewidth = 3)
    
ax.legend().set_title('Network')

plt.ylabel(r'$\min$''$(\log10$''($\lambda$))')
plt.xlabel("Size")
plt.legend(loc=2)

os.chdir('../results/figures')
plt.savefig("figure5A.svg")
plt.savefig("figure5A.png")

plt.show()
