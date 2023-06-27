# What needs to be done:
# 1) Merge each simout and xmv file
# 2) Label the attacks for #1 and #2
# 3) Show a quick plot for correctness check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

### Setup column names from MATLAB script

comps=['A','B','C','D','E','F','G','H']
column_names = ['A Feed', 
	'D Feed', 
	'E Feed', 
	'A and C Feed',
	'Recycle Flow',
	'Reactor Feed Rate',
	'Reactor Pressure',
	'Reactor Level',
	'Reactor Temperature',
	'Purge Rate',
	'Product Sep Temp',
	'Product Sep Level',
	'Product Sep Pressure',
	'Product Sep Underflow',
	'Stripper Level',
	'Stripper Pressure',
	'Stripper Underflow',
	'Stripper Temp',
	'Stripper Steam Flow',
	'Compressor Work',
	'Reactor Coolant Temp',
	'Separator Coolant Temp',
	'Comp A to Reactor',
	'Comp B to Reactor',
	'Comp C to Reactor',
	'Comp D to Reactor',
	'Comp E to Reactor',
	'Comp F to Reactor',
	'Comp A in Purge',
	'Comp B in Purge',
	'Comp C in Purge',
	'Comp D in Purge',
	'Comp E in Purge',
	'Comp F in Purge',
	'Comp G in Purge',
	'Comp H in Purge',
	'Comp D in Product',
	'Comp E in Product',
	'Comp F in Product',
	'Comp G in Product',
	'Comp H in Product',
	'D feed',
	'E Feed',
	'A Feed',
	'A and C Feed',
	'Recycle',
	'Purge',
	'Separator',
	'Stripper',
	'Steam',
	'Reactor Coolant',
	'Condenser Coolant',
	'Agitator']

hour = 2000

## Process benign dataset
sim_benign = np.loadtxt('simout.csv', delimiter=',')
xmv_benign = np.loadtxt('xmv.csv', delimiter=',')

Xbenign = np.hstack((sim_benign, xmv_benign))
df = pd.DataFrame(Xbenign, columns = column_names)
df.to_csv('TEP_train.csv', header=True, index=False)

## Add attack column, since we're done with training set
column_names.append('Atk')

## Process attack 1
sim_attack1 = np.loadtxt('simout_attack1.csv', delimiter=',')
xmv_attack1 = np.loadtxt('xmv_attack1.csv', delimiter=',')

Yattack1 = np.zeros((len(sim_attack1), 1))
Yattack1[9*hour : 10*hour] = 1
Yattack1[19*hour : 20*hour] = 1
Yattack1[29*hour : 30*hour] = 1
Yattack1[39*hour : 40*hour] = 1

Xattack1 = np.hstack((sim_attack1, xmv_attack1, Yattack1))
df1 = pd.DataFrame(Xattack1, columns = column_names)
df1.to_csv('TEP_test1.csv', header=True, index=False)

## Process attack 2
sim_attack2 = np.loadtxt('simout_attack2.csv', delimiter=',')
xmv_attack2 = np.loadtxt('xmv_attack2.csv', delimiter=',')

Yattack2 = np.zeros((len(sim_attack2), 1))
Yattack2[5*hour : 7*hour] = 1

Xattack2 = np.hstack((sim_attack2, xmv_attack2, Yattack2))
df2 = pd.DataFrame(Xattack2, columns = column_names)
df2.to_csv('TEP_test2.csv', header=True, index=False)

print('Done')