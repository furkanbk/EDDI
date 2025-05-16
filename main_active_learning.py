'''
EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE
This code implements EDDI global single ordering strategy
based on partial VAE (PNP), demonstrated on a UCI dataset.

To run this code:
python main_active_learning.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston --output_dir your_directory/model

[...docstring continues unchanged...]
'''

### load models and functions
from active_learning_functions import *
#### Import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import tensorflow_probability as tfp  # TF 2.x replacement for tf.contrib.distributions
tfd = tfp.distributions

plt.switch_backend('agg')

### load data
Data = pd.read_excel(UCI + '/d0.xls')
Data = Data.to_numpy()  # .as_matrix() deprecated, use to_numpy()

### data preprocess
max_Data = 1
min_Data = 0
Data_std = (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))
Data = Data_std * (max_Data - min_Data) + min_Data
Mask = np.ones(Data.shape)  # this UCI data is fully observed

Data_train, Data_test, mask_train, mask_test = train_test_split(
    Data, Mask, test_size=0.1, random_state=rs)

### run training and active learning
Repeat = args.repeat

p_vae_active_learning(
    Data_train, mask_train, Data_test, mask_test,
    args.epochs, args.latent_dim, args.batch_size, args.p,
    args.K, args.M, args.eval, Repeat
)

### visualize active learning
IC_RAND = np.load(args.output_dir + '/UCI_information_curve_RAND.npz')['information_curve']
IC_SING = np.load(args.output_dir + '/UCI_information_curve_SING.npz')['information_curve']
IC_CHAI = np.load(args.output_dir + '/UCI_information_curve_CHAI.npz')['information_curve']

plt.figure(0)
L = IC_SING.shape[1]
fig, ax1 = plt.subplots()

left, bottom, width, height = [0.45, 0.4, 0.45, 0.45]

if args.eval == 'rmse':
    ax1.plot(np.sqrt(IC_RAND[:, :, 0:].mean(axis=0).mean(axis=0)), 'gs', linestyle='-.', label='PNP+RAND')
    ax1.errorbar(
        np.arange(IC_RAND.shape[2]),
        np.sqrt(IC_RAND[:, :, 0:].mean(axis=0).mean(axis=0)),
        yerr=np.sqrt(IC_RAND[:, :, 0:].mean(axis=1)).std(axis=0) / np.sqrt(IC_SING.shape[0]),
        ecolor='g', fmt='gs'
    )
    ax1.plot(np.sqrt(IC_SING[:, :, 0:].mean(axis=0).mean(axis=0)), 'ms', linestyle='-.', label='PNP+SING')
    ax1.errorbar(
        np.arange(IC_SING.shape[2]),
        np.sqrt(IC_SING[:, :, 0:].mean(axis=0).mean(axis=0)),
        yerr=np.sqrt(IC_SING[:, :, 0:].mean(axis=1)).std(axis=0) / np.sqrt(IC_SING.shape[0]),
        ecolor='m', fmt='ms'
    )
    ax1.plot(np.sqrt(IC_CHAI[:, :, 0:].mean(axis=0).mean(axis=0)), 'ks', linestyle='-.', label='PNP+EDDI')
    ax1.errorbar(
        np.arange(IC_CHAI.shape[2]),
        np.sqrt(IC_CHAI[:, :, 0:].mean(axis=0).mean(axis=0)),
        yerr=np.sqrt(IC_CHAI[:, :, 0:].mean(axis=1)).std(axis=0) / np.sqrt(IC_CHAI.shape[0]),
        ecolor='k', fmt='ks'
    )

    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('avg. test RMSE', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ax1.legend(
        bbox_to_anchor=(0.0, 1.02, 1., .102), mode="expand", loc=3,
        ncol=1, borderaxespad=0., prop={'size': 20}, frameon=False
    )
    plt.show()
    plt.savefig(args.output_dir + '/PNP_all_IC_curves.png', format='png', dpi=200, bbox_inches='tight')
else:
    ax1.plot(IC_RAND[:, :, 0:].mean(axis=1).mean(axis=0), 'gs', linestyle='-.', label='PNP+RAND')
    ax1.errorbar(
        np.arange(IC_RAND.shape[2]),
        IC_RAND[:, :, 0:].mean(axis=1).mean(axis=0),
        yerr=IC_RAND[:, :, 0:].mean(axis=1).std(axis=0) / np.sqrt(IC_SING.shape[0]),
        ecolor='g', fmt='gs'
    )
    ax1.plot(IC_SING[:, :, 0:].mean(axis=1).mean(axis=0), 'ms', linestyle='-.', label='PNP+SING')
    ax1.errorbar(
        np.arange(IC_SING.shape[2]),
        IC_SING[:, :L, 0:].mean(axis=1).mean(axis=0),
        yerr=IC_SING[:, :L, 0:].mean(axis=1).std(axis=0) / np.sqrt(IC_SING.shape[0]),
        ecolor='m', fmt='ms'
    )
    ax1.plot(IC_CHAI[:, :, 0:].mean(axis=1).mean(axis=0), 'ks', linestyle='-.', label='PNP+EDDI')
    ax1.errorbar(
        np.arange(IC_CHAI.shape[2]),
        IC_CHAI[:, :L, 0:].mean(axis=1).mean(axis=0),
        yerr=IC_CHAI[:, :L, 0:].mean(axis=1).std(axis=0) / np.sqrt(IC_CHAI.shape[0]),
        ecolor='k', fmt='ks'
    )

    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('avg. neg. test likelihood', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ax1.legend(
        bbox_to_anchor=(0.0, 1.02, 1., .102), mode="expand", loc=3,
        ncol=1, borderaxespad=0., prop={'size': 20}, frameon=False
    )
    plt.show()
    plt.savefig(args.output_dir + '/PNP_all_IC_curves.png', format='png', dpi=200, bbox_inches='tight')
