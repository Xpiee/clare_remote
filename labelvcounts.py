import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

sns.set_theme()
# sns.set({'axes.grid': True}, style="ticks")
sns.set_style('ticks', {'axes.grid':'True'})
sns.set_context("paper")

labelPath = r"X:/IDEaS/MATBII/Data/New_Labels_2"

subjectsNumber = os.listdir(labelPath) # its with file extension

cnt = 1 
for subject in subjectsNumber:
    subjectLabel = os.path.join(labelPath, subject)
    dfLabel = pd.read_csv(subjectLabel)
    print(subject)
    # dfLabelCombine = pd.DataFrame(columns=['exp', 'complexity'])
    dfLabelCombine = pd.DataFrame(np.concatenate([dfLabel[['exp_0', 'com_exp_0']].values,
                                dfLabel[['exp_1', 'com_exp_1']].values,
                                dfLabel[['exp_2', 'com_exp_2']].values,
                                dfLabel[['exp_3', 'com_exp_3']].values], axis=0), columns=['exp', 'complexity'])

    defaultSeries = pd.Series({1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0, 9.0: 0}, name='exp')

    dfValues = dfLabelCombine['exp'].value_counts()

    defaultSeries.update(dfValues)

    fig, ax1 = plt.subplots(figsize=(6, 5))
    sns.barplot(x=defaultSeries.index, y=defaultSeries, color='b', ax=ax1)

    ax1.set(ylim=(0, 100))
    # ax1.legend_.set_title(None)
    ax1.set_xlabel(f'Cognitive Load Score', fontsize=15)
    ax1.set_ylabel('Counts', fontsize=15)
    # ax1.set_title('Type II', fontsize=15)
    sns.despine()
    plt.tight_layout()
    savePath = r'B:\Ideas\barplots'

    plt.savefig(os.path.join(savePath, f'Subject{cnt}.pdf'), dpi = 300)
    plt.savefig(os.path.join(savePath, f'Subject{cnt}.svg'))

    plt.show()

    cnt += 1