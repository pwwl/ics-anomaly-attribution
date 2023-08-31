"""

   Copyright 2020 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import pdb
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)

# Need to download from the SUTD drive, or ask someone who has it.
df_train = pd.read_csv("WADI_14days.csv", header=3)
df_test = pd.read_csv("WADI_attackdata.csv", header=0)

# Fix column names
new_columns = df_train.columns

# rename column names to not include "\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\"
for i in np.arange(3, len(new_columns)):
	colname = new_columns[i][46:]
	new_columns.values[i] =  colname
	df_test.columns.values[i] = colname

# Label attacks
attack_vector = np.zeros(len(df_test))

attack_indices = [
	(5139, 6624 + 1),    	# Attack 1
	(59050, 59641), 		# Attack 2 unchanged
	(60900, 62641), 		# Attack 3-4 unchanged
	(63040, 63891), 		# Attack 5 unchanged
	(70770, 71441), 		# Attack 6 unchanged
	(74895, 75591 + 1), 	# Attack 7
	(85237, 85778 + 1),		# Attack 8
	(147296, 147379 + 1),	# Attack 9
	(148656, 149489 + 1),	# Attack 10
	(149792, 150416 + 1),	# Attack 11
	(151131, 151507 + 1),	# Attack 12
	(151660, 151851 + 1),	# Attack 13
	(152167, 152746 + 1),	# Attack 14
	(163590, 164221),		# Attack 15 unchanged
	]

for (start, end) in attack_indices:
	print("Label attack from {} to {}".format(
		df_test['Time'].loc[start],
		df_test['Time'].loc[end - 1]))
	attack_vector[start:end] = 1

df_train['Attack'] = np.zeros(len(df_train))
df_test['Attack'] = attack_vector

# Patch up some NaNs
df_train['2B_AIT_004_PV'].values[61703:61713] = 486.6

df_train['2B_AIT_004_PV'].values[384154:384164] = 487.926

df_train['3_AIT_002_PV'].values[524280:524286] = 8279.1

df_train['1_AIT_002_PV'].values[623873:623879] = 0.71646

df_train['1_AIT_002_PV'].values[884845:884851] = 0.62047

df_train['1_AIT_004_PV'].values[706470:706476] = 501.642

df_train['3_AIT_004_PV'].values[974807:974818] = 1603.980

df_train.to_csv('WADI_train.csv', index=False)
df_test.to_csv('WADI_test.csv', index=False)