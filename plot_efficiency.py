import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot


matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []
    accu_train = []

    test_accu_1 = []
    test_accu_2 = []
    train_accu_1 = []
    train_accu_2 = []

    for line in open(file_name, 'r'):

        search_test_accu = re.search( r'At round (.*) validation accuracy: (.*)', line, re.M|re.I)
        if search_test_accu:
            rounds.append(int(search_test_accu.group(1)))
            accu.append(float(search_test_accu.group(2)))
            
        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M|re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, loss, accu, accu_train


accuracies = [ 
["./log_vehicle/gini_ffedavg_run1_q0",
"./log_vehicle/harmonic_ffedavg_run1_q0",
"./log_vehicle/qffedavg_run1_q0",
"./log_vehicle/qffedavg_run1_q5"
],

[
"./log_vehicle/geom_z_085_ffedavg_run1_q0",
"./log_vehicle/geom_z_050_ffedavg_run1_q0",
"./log_vehicle/qffedavg_run1_q0",
"./log_vehicle/qffedavg_run1_q5"
]
]

labels = [ 
["gini","harmonic","q=0", "q=1"],
["Geo = 0.85","Geo = 0.5","q=0", "q=1"]
]

acc_labels = list(zip(accuracies, labels))


dataset = ["Synthetic"]


plt.rcParams['figure.figsize'] = [11, 4.5]
sampling_rate=[1]


fig,axes = plt.subplots(nrows=1, ncols=len(acc_labels))
for i, pair in enumerate(acc_labels):
    acc_pair = pair[0]
    label_pair = pair[1]
    ax = axes[i]
    for acc, label in zip(acc_pair, label_pair):
        rounds0, losses0, test_accuracies0, train_accuracies0 = parse_log(acc)
        ax.errorbar(np.asarray(rounds0)[::sampling_rate[0]], np.asarray(test_accuracies0)[::sampling_rate[0]], linewidth=3.0, label=label)
        ax.set_ylabel('Testing accuracy', fontsize=22)
        ax.set_xlabel('# Rounds', fontsize=22)

        ax.legend(loc='best', frameon=False)
   
        ax.set_xlim(0, 20)

        plt.tight_layout()



plt.savefig("efficiency_vehicle.pdf")

