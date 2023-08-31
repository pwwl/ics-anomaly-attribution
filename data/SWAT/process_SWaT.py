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

import pandas as pd
import numpy as np

df_temp = pd.read_csv("SWaT_Dataset_Normal_v1.csv", header=0)
temp_columns = df_temp.columns.tolist()
# if first row is blank/not column names
if temp_columns[0] != "Timestamp" or temp_columns[0] != " Timestamp":
  df = pd.read_csv("SWaT_Dataset_Normal_v1.csv", header=1)
else:
  df = df_temp
df_test = pd.read_csv("SWaT_Dataset_Attack_v0.csv", header=0)

# Eliminate spaces from column names (namely from " Timestamp" and other random columns)
df.columns = df.columns.str.replace(' ', '')
df_test.columns = df.columns.str.replace(' ', '')

df_test['Normal/Attack'] = df_test['Normal/Attack'] != 'Normal'
df['Normal/Attack'] = False

ytest = df_test['Normal/Attack'].values
attack_idx = 0
real_ytest = np.zeros(len(ytest))

real_ytest[1738:2673] = 1  # Attack 0 (1 in Doc) on MV101
real_ytest[3046:3491] = 1  # Attack 1 (2 in Doc) on P102
real_ytest[4901:5283] = 1  # Attack 2 (3 in Doc) on LIT101

real_ytest[7233:7432] = 1  # Attack 3 (6 in Doc) on AIT202
real_ytest[7685:8113] = 1  # Attack 4 (7 in Doc) on LIT301
real_ytest[11385:12355] = 1  # Attack 5 (8 in Doc) on DPIT301

real_ytest[15361:16084] = 1  # Attack 6 (10 in Doc) on FIT401
real_ytest[90662:90917] = 1  # Attack 7 (13 in Doc) on MV304

real_ytest[93424:93705] = 1  # Attack 8 (16 in Doc) on LIT301
real_ytest[103092:103797] = 1  # Attack 8.5 (17 in Doc) on MV303

real_ytest[115822:116080] = 1  # Attack 9 (19 in Doc) on AIT504
real_ytest[116123:116515] = 1  # Attack 10 (20 in Doc) on AIT504

real_ytest[116999:117701] = 1  # Attack 11 (21 in Doc) on LIT101
real_ytest[132896:133362] = 1  # Attack 12 (22 in Doc) on UV401/AIT502
real_ytest[142927:143611] = 1  # Attack 13 (23 in Doc) on DPIT301

real_ytest[172268:172588] = 1  # Attack 14 (24 in Doc) on P203/205 
real_ytest[172892:173499] = 1  # Attack 15 (25 in Doc) on LIT401
real_ytest[198273:199716] = 1  # Attack 16 (26 in Doc) on P102/LIT301

real_ytest[227828:228362] = 1  # Attack 17 (27 in Doc) on LIT401
real_ytest[229519:263727] = 1  # Attack 18 (28 in Doc) on P302
real_ytest[280023:281185] = 1  # Attack 19 (30 in Doc) on P101/MV201/LIT101

real_ytest[302653:303020] = 1  # Attack 20 (31 in Doc) on LIT401
real_ytest[347718:348315] = 1  # Attack 21 (32 in Doc) on LIT301
real_ytest[361243:361674] = 1  # Attack 22 (33 in Doc) on LIT101

real_ytest[371519:371618] = 1  # Attack 23 (34 in Doc) on P101
real_ytest[371893:372374] = 1  # Attack 24 (35 in Doc) on P101
real_ytest[389746:390262] = 1  # Attack 25 (36 in Doc) on LIT101

real_ytest[436672:437046] = 1  # Attack 26 (37 in Doc) on FIT502
real_ytest[437455:437734] = 1  # Attack 27 (38 in Doc) on AIT402/AIT502
real_ytest[438184:438584] = 1  # Attack 28 (39 in Doc) on FIT401/AIT502
real_ytest[438659:438955] = 1  # Attack 29 (40 in Doc) on FIT401
real_ytest[443540:445191] = 1  # Attack 30 (41 in Doc) on LIT301

df_test['Normal/Attack'] = real_ytest
ytest = df_test['Normal/Attack'].values

for i in np.arange(len(df_test)):
                
  if (ytest[i] and i == 0) or (ytest[i] and not ytest[i-1]):
    print(f"Attack {attack_idx} start at {i}: {df_test['Timestamp'][i]}")

  if (ytest[i] and i == (len(df_test) - 1)) or (ytest[i] and not ytest[i+1]):
    print(f"Attack {attack_idx} end at {i}: {df_test['Timestamp'][i]}")
    attack_idx += 1

df.to_csv('SWATv0_train.csv', index=False)
df_test.to_csv('SWATv0_test.csv', index=False)
