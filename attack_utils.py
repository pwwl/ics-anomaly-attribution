"""

   Copyright 2023 Lujo Bauer, Clement Fung

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

# numpy stack
import numpy as np
import networkx as nx
import pdb

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

def get_attack_indices(dataset_name):

	if dataset_name == "BATADAL":

		attacks = [np.arange(6909, 7109), np.arange(8017, 8113), np.arange(9205, 9445), np.arange(11173, 11549), np.arange(13857, 14097), np.arange(14789, 15165), np.arange(15549, 15988)]

		true_labels = [
			["FLOW_PU9", "FLOW_PU10", "STATUS_PU9, STATUS_PU10"],
			["FLOW_PU9", "FLOW_PU10", "STATUS_PU9, STATUS_PU10"],
			["STATUS_PU1", "STATUS_PU2"],
			["FLOW_PU1", "FLOW_PU2", "STATUS_PU1", "STATUS_PU2"],
			["FLOW_PU7"],
			["FLOW_PU7"],
			["FLOW_PU7"]
		]

	elif dataset_name == "SWATv2" or dataset_name == "SWATv2-PHY":
	
		attacks = [np.arange(416, 521), np.arange(803, 1064), np.arange(1509, 1739), np.arange(2228, 2685), np.arange(3134, 3292), np.arange(3662, 4504)]
		
		true_labels = [
			["FIT 401"],
			["LIT 301"],
			["P601 Status"],
			["MV 201, P101 Status"],
			["MV 501"],
			["P301 Status"]
		]
	
	elif dataset_name == "SWAT":

		attacks = [
			np.arange(1738,2672),  # Attack 0 (1 in Doc) on MV101
			np.arange(3046,3490),  # Attack 1 (2 in Doc) on P102
			np.arange(4901,5282),  # Attack 2 (3 in Doc) on LIT101
			np.arange(7233,7431),  # Attack 3 (6 in Doc) on AIT202
			np.arange(7685,8113),  # Attack 4 (7 in Doc) on LIT301
			np.arange(11385,12355),  # Attack 5 (8 in Doc) on DPIT301
			np.arange(15361,16083),  # Attack 6 (10 in Doc) on FIT401
			np.arange(90662,90917),  # Attack 7 (13 in Doc) on MV304
			np.arange(93424,93705),  # Attack 8 (16 in Doc) on LIT301
			np.arange(103092,103797),  # Attack 8.5 (17 in Doc) on MV303

			np.arange(115822,116080),  # Attack 9 (19 in Doc) on AIT504
			np.arange(116123,116515),  # Attack 10 (20 in Doc) on AIT504
			np.arange(116999,117700),  # Attack 11 (21 in Doc) on LIT101
			np.arange(132896,133362),  # Attack 12 (22 in Doc) on UV401/AIT502
			np.arange(142927,143611),  # Attack 13 (23 in Doc) on DPIT301
			np.arange(172268,172588),  # Attack 14 (24 in Doc) on P203/205 
			np.arange(172892,173499),  # Attack 15 (25 in Doc) on LIT401
			np.arange(198273,199716),  # Attack 16 (26 in Doc) on P102/LIT301
			np.arange(227828,228361),  # Attack 17 (27 in Doc) on LIT401
			np.arange(229519,263727),  # Attack 18 (28 in Doc) on P302
			np.arange(280023,281184),  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			np.arange(302653,303019),  # Attack 20 (31 in Doc) on LIT401
			np.arange(347718,348315),  # Attack 21 (32 in Doc) on LIT301
			np.arange(361243,361674),  # Attack 22 (33 in Doc) on LIT101
			np.arange(371519,371618),  # Attack 23 (34 in Doc) on P101
			np.arange(371893,372374),  # Attack 24 (35 in Doc) on P101
			np.arange(389746,390262),  # Attack 25 (36 in Doc) on LIT101
			np.arange(436672,437046),  # Attack 26 (37 in Doc) on FIT502
			np.arange(437455,437735),  # Attack 27 (38 in Doc) on AIT402/AIT502
			np.arange(438184,438583),  # Attack 28 (39 in Doc) on FIT401/AIT502
			np.arange(438659,438955),  # Attack 29 (40 in Doc) on FIT401
			np.arange(443540,445191)  # Attack 30 (41 in Doc) on LIT301
		]

		true_labels = [
			["MV101"], # Attack 0 (1 in Doc) on MV101
			["P102"], # Attack 1 (2 in Doc) on P102
			["LIT101"], # Attack 2 (3 in Doc) on LIT101
			["AIT202"],  # Attack 3 (6 in Doc) on AIT202
			["LIT301"],  # Attack 4 (7 in Doc) on LIT301
			["DPIT301"],  # Attack 5 (8 in Doc) on DPIT301
			["FIT401"],  # Attack 6 (10 in Doc) on FIT401
			["MV304"],  # Attack 7 (13 in Doc) on MV304
			["LIT301"],  # Attack 8 (16 in Doc) on LIT301
			["MV303"],  # Attack 8.5 (17 in Doc) on LIT301
			["AIT504"],  # Attack 9 (19 in Doc) on AIT504
			["AIT504"],  # Attack 10 (20 in Doc) on AIT504
			["LIT101"],  # Attack 11 (21 in Doc) on LIT101
			["UV401", "AIT502"],  # Attack 12 (22 in Doc) on UV401/AIT502
			["DPIT301"],  # Attack 13 (23 in Doc) on DPIT301
			["P203", "P205"],  # Attack 14 (24 in Doc) on P203/205 
			["LIT401"],  # Attack 15 (25 in Doc) on LIT401
			["P101", "LIT301"],  # Attack 16 (26 in Doc) on P101/LIT301
			["LIT401"],  # Attack 17 (27 in Doc) on LIT401
			["P302"],  # Attack 18 (28 in Doc) on P302
			["P101", "MV201", "LIT101"],  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			["LIT401"],  # Attack 20 (31 in Doc) on LIT401
			["LIT301"],  # Attack 21 (32 in Doc) on LIT301
			["LIT101"],  # Attack 22 (33 in Doc) on LIT101
			["P101"],  # Attack 23 (34 in Doc) on P101
			["P101"],  # Attack 24 (35 in Doc) on P101
			["LIT101"],  # Attack 25 (36 in Doc) on LIT101
			["FIT502"],  # Attack 26 (37 in Doc) on FIT502
			["AIT402", "AIT502"],  # Attack 27 (38 in Doc) on AIT402/AIT502
			["FIT401", "AIT502"],  # Attack 28 (39 in Doc) on FIT401/AIT502
			["FIT401"],  # Attack 29 (40 in Doc) on FIT401
			["LIT301"]  # Attack 30 (41 in Doc) on LIT301
		]

		# attacks = [
		#     np.arange(1754,2693), 
		#     np.arange(3068,3510), 
		#     np.arange(4920,5302), 
		#     np.arange(6459,6848), 
		#     np.arange(7255,7450), 
		#     np.arange(7705,8133), 
		#     np.arange(11410,12373), 
		#     np.arange(15380,16100), 
		#     np.arange(73800,74520), 
		#     np.arange(90685,90917), 
		#     np.arange(92140,92570), 
		#     np.arange(93445,93720), 
		#     np.arange(103092,103808), 
		#     np.arange(115843,116101), 
		#     np.arange(116143,116537), 
		#     np.arange(117000,117720), 
		#     np.arange(132918,133380), 
		#     np.arange(142954,143650), 
		#     np.arange(172268,172588), 
		#     np.arange(172910,173521), 
		#     np.arange(198296,199740), 
		#     np.arange(227828,263727), 
		#     np.arange(279120,279240), 
		#     np.arange(280060,281230), 
		#     np.arange(302653,303019), 
		#     np.arange(347679,348279), 
		#     np.arange(361191,361634), 
		#     np.arange(371479,371579), 
		#     np.arange(371855,372335), 
		#     np.arange(389680,390219), 
		#     np.arange(436541,437009), 
		#     #np.arange(437417,437697), 
		#     np.arange(438147,438547), 
		#     np.arange(438621,438917), 
		#     np.arange(443501,445190)
		# ]

		# true_labels = [
		#     ["MV101"],
		#     ["P102"],
		#     ["LIT101"],
		#     ["MV504"],
		#     ["AIT202"],
		#     ["LIT301"],
		#     ["DPIT301"],
		#     ["FIT401"],
		#     ["MV304"],
		#     ["MV303"],
		#     ["LIT301"],
		#     ["MV303"],
		#     ["AIT504"],
		#     ["AIT504"],
		#     ["MV101", "LIT101"],
		#     ["UV401", "AIT502", "P501"],
		#     ["P602", "DPIT301", "MV502"],
		#     ["P203", "P205"],
		#     ["LIT401", "P401"],
		#     ["P101", "LIT301"],
		#     ["P302", "LIT401"],
		#     ["P302"],
		#     ["P201", "P203", "P205"],
		#     ["LIT101", "P101", "MV201"],
		#     ["LIT401"],
		#     ["LIT301"],
		#     ["LIT101"],
		#     ["P101"],
		#     ["P101", "P102"],
		#     ["LIT101"],
		#     ["P501", "FIT502"],
		#     ["AIT402", "AIT502"],
		#     ["FIT401", "AIT502"],
		#     ["FIT401"],
		#     ["LIT301"]
		# ] 

	elif dataset_name == "SWAT-PHY":

		attacks = [
			np.arange(1738,2672),  # Attack 0 (1 in Doc) on MV101
			np.arange(3046,3490),  # Attack 1 (2 in Doc) on P102
			np.arange(4901,5282),  # Attack 2 (3 in Doc) on LIT101
			#np.arange(7233,7431),  # Attack X (6 in Doc) on AIT202
			np.arange(7685,8113),  # Attack 3 (7 in Doc) on LIT301
			np.arange(11385,12355),  # Attack 4 (8 in Doc) on DPIT301
			np.arange(15361,16083),  # Attack 5 (10 in Doc) on FIT401
			np.arange(90662,90917),  # Attack 6 (13 in Doc) on MV304
			np.arange(93424,93705),  # Attack 7 (16 in Doc) on LIT301
			np.arange(103092,103797),  # Attack 8 (17 in Doc) on MV303
			#np.arange(115822,116080),  # Attack 9 (19 in Doc) on AIT504
			#np.arange(116123,116515),  # Attack 10 (20 in Doc) on AIT504
			np.arange(116999,117700),  # Attack 11 (21 in Doc) on LIT101
			np.arange(132896,133362),  # Attack 12 (22 in Doc) on UV401/AIT502
			np.arange(142927,143611),  # Attack 13 (23 in Doc) on DPIT301
			np.arange(172268,172588),  # Attack 14 (24 in Doc) on P203/205 
			np.arange(172892,173499),  # Attack 15 (25 in Doc) on LIT401
			np.arange(198273,199716),  # Attack 16 (26 in Doc) on P102/LIT301
			np.arange(227828,228361),  # Attack 17 (27 in Doc) on LIT401
			np.arange(229519,263727),  # Attack 18 (28 in Doc) on P302
			np.arange(280023,281184),  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			np.arange(302653,303019),  # Attack 20 (31 in Doc) on LIT401
			np.arange(347718,348315),  # Attack 21 (32 in Doc) on LIT301
			np.arange(361243,361674),  # Attack 22 (33 in Doc) on LIT101
			np.arange(371519,371618),  # Attack 23 (34 in Doc) on P101
			np.arange(371893,372374),  # Attack 24 (35 in Doc) on P101
			np.arange(389746,390262),  # Attack 25 (36 in Doc) on LIT101
			np.arange(436672,437046),  # Attack 26 (37 in Doc) on FIT502
			#np.arange(437455,437735),  # Attack X (38 in Doc) on AIT402/AIT502
			np.arange(438184,438583),  # Attack 27 (39 in Doc) on FIT401/AIT502
			np.arange(438659,438955),  # Attack 28 (40 in Doc) on FIT401
			np.arange(443540,445191)  # Attack 29 (41 in Doc) on LIT301
		]

		true_labels = [
			["MV101"], # Attack 0 (1 in Doc) on MV101
			["P102"], # Attack 1 (2 in Doc) on P102
			["LIT101"], # Attack 2 (3 in Doc) on LIT101
			#["AIT202"],  # Attack X (6 in Doc) on AIT202
			["LIT301"],  # Attack 3 (7 in Doc) on LIT301
			["DPIT301"],  # Attack 4 (8 in Doc) on DPIT301
			["FIT401"],  # Attack 5 (10 in Doc) on FIT401
			["MV304"],  # Attack 6 (13 in Doc) on MV304
			["LIT301"],  # Attack 7 (16 in Doc) on LIT301
			["MV303"],  # Attack 8 (17 in Doc) on LIT301
			#["AIT504"],  # Attack 9 (19 in Doc) on AIT504
			#["AIT504"],  # Attack 10 (20 in Doc) on AIT504
			["LIT101", "MV101"],  # Attack 11 (21 in Doc) on LIT101
			["UV401", "P501"],  # Attack 12 (22 in Doc) on UV401/AIT502
			["DPIT301", "MV302", "P602"],  # Attack 13 (23 in Doc) on DPIT301
			["P203"],  # Attack 14 (24 in Doc) on P203/205 
			["LIT401", "P402"],  # Attack 15 (25 in Doc) on LIT401
			["P101", "LIT301"],  # Attack 16 (26 in Doc) on P101/LIT301
			["LIT401", "P302"],  # Attack 17 (27 in Doc) on LIT401
			["P302"],  # Attack 18 (28 in Doc) on P302
			["P101", "MV201", "LIT101"],  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			["LIT401"],  # Attack 20 (31 in Doc) on LIT401
			["LIT301"],  # Attack 21 (32 in Doc) on LIT301
			["LIT101"],  # Attack 22 (33 in Doc) on LIT101
			["P101"],  # Attack 23 (34 in Doc) on P101
			["P101", "P102"],  # Attack 24 (35 in Doc) on P101
			["LIT101"],  # Attack 25 (36 in Doc) on LIT101
			["FIT502"],  # Attack 26 (37 in Doc) on FIT502
			#["AIT402", "AIT502"],  # Attack X (38 in Doc) on AIT402/AIT502
			["FIT401"],  # Attack 27 (39 in Doc) on FIT401/AIT502
			["FIT401"],  # Attack 28 (40 in Doc) on FIT401
			["LIT301"]  # Attack 29 (41 in Doc) on LIT301
		]

	elif dataset_name == "SWAT-CLEAN":

		attacks = [
			np.arange(1738,2672),  # Attack 0 (1 in Doc) on MV101
			np.arange(3046,3490),  # Attack 1 (2 in Doc) on P102
			np.arange(4901,5282),  # Attack 2 (3 in Doc) on LIT101
			#np.arange(7233,7431),  # Attack X (6 in Doc) on AIT202
			np.arange(7685,8113),  # Attack 3 (7 in Doc) on LIT301
			np.arange(11385,12355),  # Attack 4 (8 in Doc) on DPIT301
			np.arange(15361,16083),  # Attack 5 (10 in Doc) on FIT401
			np.arange(90662,90917),  # Attack 6 (13 in Doc) on MV304
			np.arange(93424,93705),  # Attack 7 (16 in Doc) on LIT301
			np.arange(103092,103797),  # Attack 8 (17 in Doc) on MV303
			np.arange(115822,116080),  # Attack 9 (19 in Doc) on AIT504
			np.arange(116123,116515),  # Attack 10 (20 in Doc) on AIT504
			np.arange(116999,117700),  # Attack 11 (21 in Doc) on LIT101
			np.arange(132896,133362),  # Attack 12 (22 in Doc) on UV401/AIT502
			np.arange(142927,143611),  # Attack 13 (23 in Doc) on DPIT301
			np.arange(172268,172588),  # Attack 14 (24 in Doc) on P203/205 
			np.arange(172892,173499),  # Attack 15 (25 in Doc) on LIT401
			np.arange(198273,199716),  # Attack 16 (26 in Doc) on P102/LIT301
			np.arange(227828,228361),  # Attack 17 (27 in Doc) on LIT401
			np.arange(229519,263727),  # Attack 18 (28 in Doc) on P302
			np.arange(280023,281184),  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			np.arange(302653,303019),  # Attack 20 (31 in Doc) on LIT401
			np.arange(347718,348315),  # Attack 21 (32 in Doc) on LIT301
			np.arange(361243,361674),  # Attack 22 (33 in Doc) on LIT101
			np.arange(371519,371618),  # Attack 23 (34 in Doc) on P101
			np.arange(371893,372374),  # Attack 24 (35 in Doc) on P101
			np.arange(389746,390262),  # Attack 25 (36 in Doc) on LIT101
			np.arange(436672,437046),  # Attack 26 (37 in Doc) on FIT502
			#np.arange(437455,437735),  # Attack X (38 in Doc) on AIT402/AIT502
			np.arange(438184,438583),  # Attack 27 (39 in Doc) on FIT401/AIT502
			np.arange(438659,438955),  # Attack 28 (40 in Doc) on FIT401
			np.arange(443540,445191)  # Attack 29 (41 in Doc) on LIT301
		]

		true_labels = [
			["MV101"], # Attack 0 (1 in Doc) on MV101
			["P102"], # Attack 1 (2 in Doc) on P102
			["LIT101"], # Attack 2 (3 in Doc) on LIT101
			#["AIT202"],  # Attack X (6 in Doc) on AIT202
			["LIT301"],  # Attack 3 (7 in Doc) on LIT301
			["DPIT301"],  # Attack 4 (8 in Doc) on DPIT301
			["FIT401"],  # Attack 5 (10 in Doc) on FIT401
			["MV304"],  # Attack 6 (13 in Doc) on MV304
			["LIT301"],  # Attack 7 (16 in Doc) on LIT301
			["MV303"],  # Attack 8 (17 in Doc) on LIT301
			["AIT504"],  # Attack 9 (19 in Doc) on AIT504
			["AIT504"],  # Attack 10 (20 in Doc) on AIT504
			["LIT101", "MV101"],  # Attack 11 (21 in Doc) on LIT101
			["UV401", "P501"],  # Attack 12 (22 in Doc) on UV401/AIT502
			["DPIT301", "MV302", "P602"],  # Attack 13 (23 in Doc) on DPIT301
			["P203"],  # Attack 14 (24 in Doc) on P203/205 
			["LIT401", "P402"],  # Attack 15 (25 in Doc) on LIT401
			["P101", "LIT301"],  # Attack 16 (26 in Doc) on P101/LIT301
			["LIT401", "P302"],  # Attack 17 (27 in Doc) on LIT401
			["P302"],  # Attack 18 (28 in Doc) on P302
			["P101", "MV201", "LIT101"],  # Attack 19 (30 in Doc) on P101/MV201/LIT101
			["LIT401"],  # Attack 20 (31 in Doc) on LIT401
			["LIT301"],  # Attack 21 (32 in Doc) on LIT301
			["LIT101"],  # Attack 22 (33 in Doc) on LIT101
			["P101"],  # Attack 23 (34 in Doc) on P101
			["P101", "P102"],  # Attack 24 (35 in Doc) on P101
			["LIT101"],  # Attack 25 (36 in Doc) on LIT101
			["FIT502"],  # Attack 26 (37 in Doc) on FIT502
			#["AIT402", "AIT502"],  # Attack X (38 in Doc) on AIT402/AIT502
			["FIT401"],  # Attack 27 (39 in Doc) on FIT401/AIT502
			["FIT401"],  # Attack 28 (40 in Doc) on FIT401
			["LIT301"]  # Attack 29 (41 in Doc) on LIT301
		]

	elif dataset_name == "WADI" or dataset_name == "WADI-CLEAN" or dataset_name == 'WADI-PHY':

		attacks = [
			np.arange(5139, 6619),       # Attack 1
			np.arange(59069, 59613),     # Attack 2 
			np.arange(61058, 61622),     # Attack 3
			np.arange(61667, 61936),     # Attack 4
			np.arange(63046, 63891),     # Attack 5
			np.arange(70795, 71458),     # Attack 6
			np.arange(74828, 75592),     # Attack 7
			np.arange(85239, 85779),     # Attack 8
			np.arange(147297, 147380),   # Attack 9
			np.arange(148657, 149479),   # Attack 10
			np.arange(149793, 150417),   # Attack 11
			np.arange(151132, 151508),   # Attack 12
			np.arange(151661, 151853),   # Attack 13
			np.arange(152174, 152742),   # Attack 14
			np.arange(163804, 164221)    # Attack 15
		]

		true_labels = [
			["1_MV_001_STATUS"],       # Attack 1
			["1_FIT_001_PV"],     # Attack 2 
			["2_MV_003_STATUS"],     # Attack 3
			["1_AIT_001_PV"],     # Attack 4
			["2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO", "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO"],     # Attack 5
			["2_FIC_101_PV", "2_FIC_201_PV"],     # Attack 6
			["1_AIT_002_PV", "2_MV_003_STATUS"],     # Attack 7
			["2_MCV_007_CO"],     # Attack 8
			["1_P_006_STATUS"],   # Attack 9
			["1_MV_001_STATUS"],   # Attack 10
			["2_MCV_007_CO"],   # Attack 11
			["2_MCV_007_CO"],   # Attack 12
			["2_PIC_003_CO", "2_PIC_003_SP"],   # Attack 13
			["1_P_001_STATUS", "1_P_003_STATUS"],   # Attack 14
			["2_MV_003_STATUS"]    # Attack 15
		]

	elif dataset_name == "TEP":

		attacks = [
			np.arange(18000, 20000),       # Attack 1
			np.arange(38000, 40000),     # Attack 2 
			np.arange(58000, 60000),     # Attack 3
			np.arange(78000, 80000)     # Attack 4
		]

		true_labels = [
			["Reactor Temperature"],
			["Reactor Temperature"],
			["Reactor Temperature"],
			["Reactor Temperature"]
		]

	elif dataset_name == "TEP2":

		attacks = [
			np.arange(10000, 14000),       
		]

		true_labels = [
			["Stripper Level"]
		]

	elif dataset_name == "TEPK-raw":

		attacks = [
			np.arange(71398, 71425),
			np.arange(38571, 47637),
			np.arange(59455, 69454),
			np.arange(37449, 105705)
		]

		true_labels = [
			["Reactor Temperature"],
			["Stripper (MV)", "Stripper Level", "Stripper Underflow"],
			["D Feed (MV)"],
			["A and C Feed (MV)", "Purge (MV)", "Steam (MV)", "Stripper Underflow"]
		]

	elif dataset_name == 'TEPK':

		attacks = [
			np.arange(71398, 71425),
			np.arange(38571, 41644),
			np.arange(41645, 44667),
			np.arange(44666, 47687),
			np.arange(59455, 69454),
			np.arange(37449, 105705)
		]

		true_labels = [
			["Reactor Temperature"],
			["Stripper (MV)"],
			["Stripper Level"],
			["Stripper Underflow"],
			["D Feed (MV)"],
			["A and C Feed (MV)", "Purge (MV)", "Steam (MV)", "Stripper Underflow"]
		]

	else:

		print(f'Warning: dataset {dataset_name} does not exist.')
		attacks = []
		true_labels = []

	return attacks, true_labels

def get_attack_sds(dataset_name):

	sds = []

	if dataset_name == 'SWAT-CLEAN':

		sds = [
			(0, "MV101", 'cons', 0.61), # Attack 0 (1 in Doc) on MV101
			(1, "P102", 'cons', 100), # Attack 1 (2 in Doc) on P102
			(2, "LIT101", 'line', 2.77), # Attack 2 (3 in Doc) on LIT101
			(3 ,"LIT301", 'cons', 3.17),  # Attack 3 (7 in Doc) on LIT301
			(4, "DPIT301", 'cons', 4.20),  # Attack 4 (8 in Doc) on DPIT301
			(5, "FIT401", 'cons', -17),  # Attack 5 (10 in Doc) on FIT401
			(6, "MV304", 'cons', -0.1),  # Attack 6 (13 in Doc) on MV304
			(7, "LIT301", 'line', -3.38),  # Attack 7 (16 in Doc) on LIT301
			(8, "MV303", 'cons', -0.12),  # Attack 8 (17 in Doc) on LIT301
			(9, "AIT504", 'cons', 0.58),  # Attack 9 (19 in Doc) on AIT504
			(10, "AIT504", 'cons', 36.31),  # Attack 10 (20 in Doc) on AIT504
			(11, "LIT101", 'cons', 0.92),  # Attack 11 (21 in Doc) on LIT101/MV101
			(11, "MV101", 'cons', 0.61),  # Attack 11 (21 in Doc) on LIT101/MV101
			(12, "UV401", 'cons', -17.64),  # Attack 12 (22 in Doc) on UV401/AIT502
			(12, "P501", 'cons', -17.19),  # Attack 12 (22 in Doc) on UV401/AIT502
			(13, "DPIT301", 'cons', -2.39),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(13, "MV302", 'cons', 0.48),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(13, "P602", 'cons', -0.09),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(14, "P203", 'cons', -1.72),  # Attack 14 (24 in Doc) on P203/205 
			(15, "LIT401", 'cons', 1.32), # Attack 15 (25 in Doc) on LIT401/P402
			(15, "P402", 'cons', 0.06), # Attack 15 (25 in Doc) on LIT401/P402
			(16, "P101", 'cons', 0.58),  # Attack 16 (26 in Doc) on P101/LIT301
			(16, "LIT301", 'cons', -1.04),  # Attack 16 (26 in Doc) on P101/LIT301
			(17, "LIT401", 'cons', -3.19),  # Attack 17 (27 in Doc) on LIT401/P302
			(17, "P302", 'cons', 0.47),  # Attack 17 (27 in Doc) on LIT401/P302
			(18, "P302", 'cons', -2.14),   # Attack 18 (28 in Doc) on P302
			(19, "P101", 'cons', 0.58), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(19, "MV201", 'cons', 0.57), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(19, "LIT101", 'cons', 0.92), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(20, "LIT401", 'cons', -3.19),  # Attack 20 (31 in Doc) on LIT401
			(21, "LIT301", 'cons', 3.18),  # Attack 21 (32 in Doc) on LIT301
			(22, "LIT101", 'cons', 1.75),  # Attack 22 (33 in Doc) on LIT101
			(23, "P101", 'cons', -1.72),  # Attack 23 (34 in Doc) on P101
			(24, "P101", 'cons', -1.72),  # Attack 24 (35 in Doc) on P101
			(24, "P102", 'cons', 1e-3),  # Attack 24 (35 in Doc) on P101
			(25, "LIT101", 'cons', -2.82),  # Attack 25 (36 in Doc) on LIT101
			(26, "FIT502", 'cons', 0.25),  # Attack 26 (37 in Doc) on FIT502
			(27, "FIT401", 'cons', -12),  # Attack 27 (39 in Doc) on FIT401/AIT502
			(28, "FIT401", 'cons', -17),  # Attack 28 (40 in Doc) on FIT401
			(29, "LIT301", 'line', -5.66)  # Attack 29 (41 in Doc) on LIT301
		]

	elif dataset_name == 'SWAT-PHY':

		sds = [
			(0, "MV101", 'cons', 0.61), # Attack 0 (1 in Doc) on MV101
			(1, "P102", 'cons', 100), # Attack 1 (2 in Doc) on P102
			(2, "LIT101", 'line', 2.77), # Attack 2 (3 in Doc) on LIT101
			#(3, "AIT202", 'cons', 26.56), # Attack X (3 in Doc) on AIT202
			(3 ,"LIT301", 'cons', 3.17),  # Attack 3 (7 in Doc) on LIT301
			(4, "DPIT301", 'cons', 4.20),  # Attack 4 (8 in Doc) on DPIT301
			(5, "FIT401", 'cons', -17),  # Attack 5 (10 in Doc) on FIT401
			(6, "MV304", 'cons', -0.1),  # Attack 6 (13 in Doc) on MV304
			(7, "LIT301", 'line', -3.38),  # Attack 7 (16 in Doc) on LIT301
			(8, "MV303", 'cons', -0.12),  # Attack 8 (17 in Doc) on LIT301
			#(10, "AIT504", 'cons', 0.58),  # Attack 9 (19 in Doc) on AIT504
			#(11, "AIT504", 'cons', 36.31),  # Attack 10 (20 in Doc) on AIT504
			(9, "LIT101", 'cons', 0.92),  # Attack 11 (21 in Doc) on LIT101/MV101
			(9, "MV101", 'cons', 0.61),  # Attack 11 (21 in Doc) on LIT101/MV101
			(10, "UV401", 'cons', -17.64),  # Attack 12 (22 in Doc) on UV401/AIT502
			(10, "P501", 'cons', -17.19),  # Attack 12 (22 in Doc) on UV401/AIT502
			(11, "DPIT301", 'cons', -2.39),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(11, "MV302", 'cons', 0.48),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(11, "P602", 'cons', -0.09),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(12, "P203", 'cons', -1.72),  # Attack 14 (24 in Doc) on P203/205 
			(13, "LIT401", 'cons', 1.32), # Attack 15 (25 in Doc) on LIT401/P402
			(13, "P402", 'cons', 0.06), # Attack 15 (25 in Doc) on LIT401/P402
			(14, "P101", 'cons', 0.58),  # Attack 16 (26 in Doc) on P101/LIT301
			(14, "LIT301", 'cons', -1.04),  # Attack 16 (26 in Doc) on P101/LIT301
			(15, "LIT401", 'cons', -3.19),  # Attack 17 (27 in Doc) on LIT401/P302
			(15, "P302", 'cons', 0.47),  # Attack 17 (27 in Doc) on LIT401/P302
			(16, "P302", 'cons', -2.14),   # Attack 18 (28 in Doc) on P302
			(17, "P101", 'cons', 0.58), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(17, "MV201", 'cons', 0.57), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(17, "LIT101", 'cons', 0.92), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(18, "LIT401", 'cons', -3.19),  # Attack 20 (31 in Doc) on LIT401
			(19, "LIT301", 'cons', 3.18),  # Attack 21 (32 in Doc) on LIT301
			(20, "LIT101", 'cons', 1.75),  # Attack 22 (33 in Doc) on LIT101
			(21, "P101", 'cons', -1.72),  # Attack 23 (34 in Doc) on P101
			(22, "P101", 'cons', -1.72),  # Attack 24 (35 in Doc) on P101
			(22, "P102", 'cons', 1e-3),  # Attack 24 (35 in Doc) on P101
			(23, "LIT101", 'cons', -2.82),  # Attack 25 (36 in Doc) on LIT101
			(24, "FIT502", 'cons', 0.25),  # Attack 26 (37 in Doc) on FIT502
			#(28, "AIT402", 'cons', 6.88),  # Attack 26.5 (38 in Doc) on AIT402
			#(28, "AIT502", 'cons', 7.52),  # Attack 26.5 (38 in Doc) on AIT502
			(25, "FIT401", 'cons', -12),  # Attack 27 (39 in Doc) on FIT401/AIT502
			(26, "FIT401", 'cons', -17),  # Attack 28 (40 in Doc) on FIT401
			(27, "LIT301", 'line', -5.66)  # Attack 29 (41 in Doc) on LIT301
		]

	elif dataset_name == 'SWAT':

		sds = [
			(0, "MV101", 'cons', 'solo', 0.61), # Attack 0 (1 in Doc) on MV101
			(1, "P102", 'cons', 'solo', 100), # Attack 1 (2 in Doc) on P102
			(2, "LIT101", 'line', 'solo', 2.77), # Attack 2 (3 in Doc) on LIT101
			(3, "AIT202", 'cons', 'solo', 26.56), # Attack X (3 in Doc) on AIT202
			(4 ,"LIT301", 'cons', 'solo', 3.17),  # Attack 3 (7 in Doc) on LIT301
			(5, "DPIT301", 'cons', 'solo', 4.20),  # Attack 4 (8 in Doc) on DPIT301
			(6, "FIT401", 'cons', 'solo', -17),  # Attack 5 (10 in Doc) on FIT401
			(7, "MV304", 'cons', 'solo', -0.1),  # Attack 6 (13 in Doc) on MV304
			(8, "LIT301", 'line', 'solo', -3.38),  # Attack 7 (16 in Doc) on LIT301
			(9, "MV303", 'cons', 'solo', -0.12),  # Attack 8 (17 in Doc) on LIT301
			(10, "AIT504", 'cons', 'solo', 0.58),  # Attack 9 (19 in Doc) on AIT504
			(11, "AIT504", 'cons', 'solo', 36.31),  # Attack 10 (20 in Doc) on AIT504
			(12, "LIT101", 'cons','multi',  0.92),  # Attack 11 (21 in Doc) on LIT101/MV101
			(12, "MV101", 'cons', 'multi', 0.61),  # Attack 11 (21 in Doc) on LIT101/MV101
			(13, "UV401", 'cons', 'multi', -17.64),  # Attack 12 (22 in Doc) on UV401/AIT502/P501
			(13, "P501", 'cons', 'multi', -17.19),  # Attack 12 (22 in Doc) on UV401/AIT502/P501
			(14, "DPIT301", 'cons', 'multi', -2.39),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(14, "MV302", 'cons', 'multi', 0.48),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(14, "P602", 'cons', 'multi', -0.09),  # Attack 13 (23 in Doc) on DPIT301/MV302/P602
			(15, "P203", 'cons', 'solo', -1.72),  # Attack 14 (24 in Doc) on P203/205 
			(16, "LIT401", 'cons', 'multi', 1.32), # Attack 15 (25 in Doc) on LIT401/P402
			(16, "P402", 'cons', 'multi', 0.06), # Attack 15 (25 in Doc) on LIT401/P402
			(17, "P101", 'cons', 'multi', 0.58),  # Attack 16 (26 in Doc) on P101/LIT301
			(17, "LIT301", 'cons', 'multi', -1.04),  # Attack 16 (26 in Doc) on P101/LIT301
			(18, "LIT401", 'cons', 'multi', -3.19),  # Attack 17 (27 in Doc) on LIT401/P302
			(18, "P302", 'cons', 'multi', 0.47),  # Attack 17 (27 in Doc) on LIT401/P302
			(19, "P302", 'cons', 'solo', -2.14),   # Attack 18 (28 in Doc) on P302
			(20, "P101", 'cons', 'multi', 0.58), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(20, "MV201", 'cons', 'multi', 0.57), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(20, "LIT101", 'cons', 'multi', 0.92), # Attack 19 (30 in Doc) on P101/MV201/LIT101
			(21, "LIT401", 'cons', 'solo', -3.19),  # Attack 20 (31 in Doc) on LIT401
			(22, "LIT301", 'cons', 'solo', 3.18),  # Attack 21 (32 in Doc) on LIT301
			(23, "LIT101", 'cons', 'solo', 1.75),  # Attack 22 (33 in Doc) on LIT101
			(24, "P101", 'cons', 'solo', -1.72),  # Attack 23 (34 in Doc) on P101
			(25, "P101", 'cons', 'multi', -1.72),  # Attack 24 (35 in Doc) on P101/P102
			(25, "P102", 'cons', 'multi', 1e-3),  # Attack 24 (35 in Doc) on P101/P102
			(26, "LIT101", 'cons', 'solo', -2.82),  # Attack 25 (36 in Doc) on LIT101
			(27, "FIT502", 'cons', 'solo', 0.25),  # Attack 26 (37 in Doc) on FIT502
			(28, "AIT402", 'cons', 'multi', 6.88),  # Attack 26.5 (38 in Doc) on AIT402/AIT502
			(28, "AIT502", 'cons', 'multi', 7.52),  # Attack 26.5 (38 in Doc) on AIT502/AIT502
			(29, "FIT401", 'cons', 'solo', -12),  # Attack 27 (39 in Doc) on FIT401/AIT502
			(30, "FIT401", 'cons', 'solo', -17),  # Attack 28 (40 in Doc) on FIT401
			(31, "LIT301", 'line', 'solo', -5.66)  # Attack 29 (41 in Doc) on LIT301
		]

	elif dataset_name == 'TEPK':

		sds = [
			(0, "Reactor Temperature", 'cons', 50),
			(1, "Stripper (MV)", 'cons', -0.08),
			(1, "Stripper Level", 'cons', -0.03),
			(1, "Stripper Underflow", 'cons', 0.95),
			(2, "D Feed (MV)", 'cons', 0.27)
		]

	elif dataset_name == 'WADI':

		sds = [
			(0, '1_MV_001_STATUS', 'cons', 'solo', 1.62),
			(1, '1_FIT_001_PV', 'cons', 'solo', 1.27),
			(2, '2_MV_003_STATUS', 'cons', 'solo', 0.50), 
			(3, '1_AIT_001_PV', 'cons', 'solo', 35.348),
			(4, '2_MCV_101_CO', 'cons', 'multi', 5.83),
			(4, '2_MCV_201_CO', 'cons', 'multi', 5.16),
			(4, '2_MCV_301_CO', 'cons', 'multi', 3.94),
			(4, '2_MCV_401_CO', 'cons', 'multi', 5.68),
			(4, '2_MCV_501_CO', 'cons', 'multi', 5.1),
			(4, '2_MCV_601_CO', 'cons', 'multi', 3.61),
			(5, '2_FIC_101_PV', 'cons', 'multi', 0.903),
			(5, '2_FIC_201_PV', 'cons', 'multi', 1.298),
			(6, '1_AIT_002_PV', 'cons', 'multi', 91),
			(6, '2_MV_003_STATUS', 'cons', 'multi', 1.79),
			(7, '2_MCV_007_CO', 'cons', 'solo', 100),
			(8, '1_P_006_STATUS', 'cons', 'solo', 100),
			(9, '1_MV_001_STATUS', 'cons', 'solo', 1.626),
			(10, '2_MCV_007_CO', 'cons', 'solo', 100),
			(11, '2_MCV_007_CO', 'cons', 'solo', 100),
			(12, '2_PIC_003_CO', 'cons', 'multi', 2.94),
			(12, '2_PIC_003_SP', 'cons', 'multi', 100),
			(13, '1_P_001_STATUS', 'cons', 'multi', 0.615),
			(13, '1_P_003_STATUS', 'cons', 'multi', 0.615),
			(14, '2_MV_003_STATUS', 'cons', 'solo', 1.79),
		]

	elif dataset_name == 'WADI-PHY':

		sds = [
			(0, '1_MV_001_STATUS', 'cons', 1.62),
			(1, '1_FIT_001_PV', 'cons', 1.27),
			(2, '2_MV_003_STATUS', 'cons', 0.50), 
			#(3, '1_AIT_001_PV', 'cons', 35.348),
			(4, '2_MCV_101_CO', 'cons', 5.83),
			(4, '2_MCV_201_CO', 'cons', 5.16),
			(4, '2_MCV_301_CO', 'cons', 3.94),
			(4, '2_MCV_401_CO', 'cons', 5.68),
			(4, '2_MCV_501_CO', 'cons', 5.1),
			(4, '2_MCV_601_CO', 'cons', 3.61),
			(5, '2_FIC_101_PV', 'cons', 0.903),
			(5, '2_FIC_201_PV', 'cons', 1.298),
			#(6, '1_AIT_002_PV', 'cons', 91),
			(6, '2_MV_003_STATUS', 'cons', 1.79),
			(7, '2_MCV_007_CO', 'cons', 100),
			(8, '1_P_006_STATUS', 'cons', 100),
			(8, '1_MV_001_STATUS', 'cons', 1.626),
			(10, '2_MCV_007_CO', 'cons', 100),
			(11, '2_MCV_007_CO', 'cons', 100),
			(12, '2_PIC_003_CO', 'cons', 100),
			(13, '1_P_001_STATUS', 'cons', 0.615),
			(13, '1_P_003_STATUS', 'cons', 0.615),
			(14, '2_MV_003_STATUS', 'cons', 1.79),
		]


	return sds

def get_sensor_subsets(dataset_name, by_plc = True):

	subprocess_idxs = []
	subprocess_labels = []

	if dataset_name == 'SWAT':

		if by_plc:

			# Which sub process of SWaT?
			subprocess_idxs = [
				[0, 1, 2, 3],                   # PLC 1
				[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # PLC 2
				[9, 16, 17, 18, 19, 20, 23],    # PLC 3
				[26, 27, 28, 29],               # PLC 4
				[34, 38, 39, 42],               # PLC 5
			]

			subprocess_labels =[
				['FIT101', 'LIT101', 'MV101', 'P101'],
				['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
				['MV201', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'P301'],
				['AIT402', 'FIT401', 'LIT401', 'P401'],
				['AIT501', 'FIT501', 'FIT502', 'P501'],
			]

		else:

			subprocess_idxs = [
				range(0, 5),   # Process 1
				range(5, 16),  # Process 2
				range(16, 25), # Process 3
				range(25, 34), # Process 4
				range(34, 47), # Process 5
				range(47, 51), # Process 6
			]

	elif dataset_name == 'TEP' or dataset_name == 'TEPK':
		
		if by_plc:

			subprocess_idxs = [
				[1, 16, 39, 41],     # XMV 1
				[2, 16, 39, 42],     # XMV 2
				[0, 16, 22, 24, 43], # XMV 3
				[3, 16, 22, 24, 44], # XMV 4
				[6, 9, 16, 46],      # XMV 6
				[11, 13, 16, 47],    # XMV 7
				[14, 16, 48],        # XMV 8
				[8, 50],             # XMV 10
				[7, 10, 51],         # XMV 11
				
				# Physical
				[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 16, 22, 24, 39, 41, 42, 43, 44, 46, 47, 48, 50, 51],
			]

			subprocess_labels = [
				['D Feed', 'Stripper Underflow', 'Comp G in Product', 'D Feed (MV)'],
				['E Feed', 'Stripper Underflow', 'Comp G in Product', 'E Feed (MV)'],
				['A Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A Feed (MV)'],
				['A and C Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A and C Feed (MV)'],
				['Reactor Pressure', 'Purge Rate', 'Stripper Underflow', 'Purge (MV)'],
				['Product Sep Level', 'Product Sep Underflow', 'Stripper Underflow', 'Separator (MV)'],
				['Stripper Level', 'Stripper Underflow', 'Stripper (MV)'],
				['Reactor Temperature', 'Reactor Coolant (MV)'],
				['Reactor Level', 'Product Sep Temp', 'Condenser Coolant (MV)'],
				['Physical']
			]

	
	return subprocess_idxs, subprocess_labels

SWAT_SUB_MAP = {
	'1_Raw_Water_Tank' : ['MV101', 'LIT101', 'FIT101', 'P101', 'P102'],
	'2_Chemical' : ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT201', 'AIT202', 'AIT203', 'MV201'], 
	#'2_Chemical' : ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT202', 'AIT203', 'MV201'], # if AIT201 causes too much bias
	'3_UltraFilt' : ['FIT301', 'LIT301', 'DPIT301', 'P301', 'P302', 'MV301', 'MV302', 'MV303', 'MV304'],
	'4_DeChloro' : ['UV401', 'P401', 'P402', 'P403', 'P404', 'AIT401', 'AIT402', 'FIT401', 'LIT401'],
	'5_RO' : ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503'],
	'6_Return' : ['P601', 'P602', 'P603', 'FIT601']
}

WADI_SUB_MAP = {
	'1_Raw_Water_Tank' : ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV',
		'1_FIT_001_PV', '1_LS_001_AL', '1_LS_002_AL', '1_LT_001_PV',
		'1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS',
		'1_P_001_STATUS', '1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS', '1_P_005_STATUS', '1_P_006_STATUS'],
	'Elevated' : ['2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '2_LT_001_PV', '2_LT_002_PV', '2_PIT_001_PV',
	 	'2_MV_001_STATUS', '2_MV_002_STATUS', '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS',
		'2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV',],
	'Booster': ['2_DPIT_001_PV', '2_MCV_007_CO', '2_MV_009_STATUS', 
		'2_P_003_SPEED', '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS',
		'2_PIT_002_PV', '2_PIT_003_PV', '2B_AIT_001_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
		'2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP'],
	'Consumers': ['2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', '2_FIC_301_SP', 
		'2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP',
		'2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', 
		'2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', 
		'2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL',
		'2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO',
		'2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS', '2_MV_601_STATUS',
		'2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS'],
	'Return': ['3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', 
		'3_FIT_001_PV', '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS']
}

def is_actuator(dataset, label):
	
	if dataset == 'SWAT':
		if 'IT' in label:
			return False
		else:
			return True
	elif dataset == 'WADI':
		if 'STATUS' in label:
			return True
		else:
			return False
	elif dataset == 'TEP':
		if label[0] == 'a':
			return True
		else:
			return False
	
	return False

# Given a feature, return a vector of ranking outcomes
def get_rel_scores(dataset, sensor_cols, graph, true_col_name):

	max_dist = 5
	rel_scores = np.zeros(len(sensor_cols))
	distances = nx.shortest_path_length(graph, true_col_name)

	for i in range(len(sensor_cols)):
		col_name = sensor_cols[i]
		
		if col_name in distances:
			rel_scores[i] = 5 - min(distances[col_name], 5)
		elif col_to_subsystem_idx(dataset, col_name) == col_to_subsystem_idx(dataset, true_col_name):
			rel_scores[i] = 1
		else:
			rel_scores[i] = 0

	return rel_scores


def col_to_subsystem_idx(dataset, col_name):
	true_idx = -1
	if dataset == 'SWAT':
		true_idx = int(col_name[-3]) - 1
	elif dataset == 'WADI':
		sub_map = WADI_SUB_MAP
		for index, (key, val) in enumerate(sub_map.items()):
			if col_name in sub_map[key]:
				true_idx = index
				break
	
	return true_idx


def to_subsystem_scores(dataset, sensor_cols, flat_scores):

	sub_idxs = []
	sub_errors = []
	sub_error_map = dict()

	if dataset == 'SWAT':

		sub_map = SWAT_SUB_MAP
		for key in sub_map.keys():
			for item in sub_map[key]:
				sub_idxs.append(sensor_cols.index(item))
			sub_errors.append(np.mean(flat_scores[sub_idxs]))
			sub_error_map[key] = np.mean(flat_scores[sub_idxs])

	elif dataset == 'WADI':

		sub_map = WADI_SUB_MAP
		for key in sub_map.keys():
			for item in sub_map[key]:
				sub_idxs.append(sensor_cols.index(item))
			sub_errors.append(np.mean(flat_scores[sub_idxs]))
			sub_error_map[key] = np.mean(flat_scores[sub_idxs])

	return sub_errors

def subsample(data, num_to_sample):

	shuffle_idx = np.random.permutation(len(data))[:num_to_sample]
	return data[shuffle_idx]

