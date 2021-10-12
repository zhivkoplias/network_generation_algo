"""
simple script for plotting the results of a stability test stroed in a csv file. 
Note that the column names are assumed to be constant as they are set in a previous script 
"""

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np 
import sys

input = sys.argv[1]

df = pd.read_csv(input,sep=",")

df["log lambda"] = np.log10(abs(df["min jtlide_lambda"].values))

sns.lineplot(x="size", y="log lambda",hue="Method",data=df)
plt.ylabel("log10(abs(jtilde_lambda))")
plt.xlabel("Size")

if len(sys.argv) > 2:
    plt.savefig(sys.argv[2])
else:
    plt.show()
