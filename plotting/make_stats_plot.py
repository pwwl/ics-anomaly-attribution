
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib
import scipy.stats as ss

import sys
sys.path.append('..')

matplotlib.rcParams['pdf.fonttype'] = 42

def load_stats_obj(header, model_type):

	ncols = [51, 119, 53]
	datasets = ['SWAT', 'WADI', 'TEP']
	run_name = 'results_ns1'

	methods = [
		['MSE', 'mse_ranking'],
		['SM', 'smap_ranking'],
		['SHAP', 'shap_ranking'],
		['LEMNA', 'lemna_ranking'],
	]

	all_dfs = []

	for dx in range(len(datasets)):

		dataset = datasets[dx] 

		# Load method df
		if model_type == 'CNN':
			lookup_name = f'{model_type}-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model_type}-{dataset}-l2-hist50-units64-{run_name}'

		df = pickle.load(open(f'meta-storage/model-detection-ranks/{header}-{lookup_name}.pkl', 'rb'))

		# Add normalization to method rankings
		for _, col_name in methods:
			df[f'{col_name}_norm'] = df[f'{col_name}'] / ncols[dx]

		all_dfs.append(df)

	full_df = pd.concat(all_dfs, ignore_index=True)

	return full_df, methods

def main(model_type, detection_type='real'):

	if detection_type == 'ideal':
		full_df, methods = load_stats_obj('idealdet', model_type)
	elif detection_type == 'real':
		full_df, methods = load_stats_obj('realdet', model_type)

	for method, dep_var in methods: 

		dep_var_norm = f'{dep_var}_norm'
		#dep_var_norm = f'{dep_var}'
		dep_var_norm_idx = np.where(full_df.columns == dep_var_norm)[0][0]

		print(f'{model_type} {method} SD magnitude')
		st, pval = ss.pearsonr(np.log(full_df['sd'].values), full_df[dep_var_norm].values)
		plt.title(f'{method} AvgRank ~ log(manipulation magnitude)\n corr={st:.3f} pval={pval:.8f}')
		plt.xlabel('log(manipulation magnitude)')
		plt.ylabel(f'Normalized {method} AvgRank')
		print(f'Pearson corr={st} pval={pval}')
		plt.scatter(np.log(full_df['sd'].values), full_df[dep_var_norm].values)
		#plt.show()
		plt.close()

		# 3 group measure
		sen_idx = np.where(full_df['sensor_type'] == 'Sensor')[0]
		act_float_idx = np.where((full_df['sensor_type'] == 'Actuator') & (full_df['val_type'] == 'float'))[0]
		act_bool_idx = np.where((full_df['sensor_type'] == 'Actuator') & (full_df['val_type'] == 'bool'))[0]
		print(f'Three group type: sen {np.mean(full_df.iloc[sen_idx, dep_var_norm_idx])} vs act-float {np.mean(full_df.iloc[act_float_idx, dep_var_norm_idx])} vs act-bool {np.mean(full_df.iloc[act_bool_idx, dep_var_norm_idx])}')
		stat, pval = ss.f_oneway(full_df.iloc[sen_idx, dep_var_norm_idx], full_df.iloc[act_float_idx, dep_var_norm_idx], full_df.iloc[act_bool_idx, dep_var_norm_idx])
		print(f'{method} ANOVA={stat:.3f} pval={pval:.5f}')
		
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))
		ax.set_title(f'Normalized {method} AvgRank by feature type \nANOVA={stat:.3f} pval={pval:.5f}')
		ax.boxplot([full_df.iloc[sen_idx, dep_var_norm_idx], full_df.iloc[act_float_idx, dep_var_norm_idx], full_df.iloc[act_bool_idx, dep_var_norm_idx]], vert=False)
		ax.set_xlim([0,1])	
		ax.set_yticks([1,2,3])
		ax.set_yticklabels(['Sensor', 'Continuous\nActuator', 'Categorical\nActuator'])
		
		fig.tight_layout()
		plt.savefig(f'{method}-boxplot.pdf')
		plt.close()

		solo_idx = np.where(full_df['is_multi'] == 'solo')[0]
		multi_idx = np.where(full_df['is_multi'] == 'multi')[0]
		print(f'Solo vs multi: {np.mean(full_df.iloc[solo_idx, dep_var_norm_idx])} vs {np.mean(full_df.iloc[multi_idx, dep_var_norm_idx])}')

		stat, pval = ss.f_oneway(full_df.iloc[solo_idx, dep_var_norm_idx], full_df.iloc[multi_idx, dep_var_norm_idx])
		print(f'{method} ANOVA={stat:.3f} pval={pval:.5f}')
		plt.boxplot([full_df.iloc[solo_idx, dep_var_norm_idx], full_df.iloc[multi_idx, dep_var_norm_idx]], vert=False)
		plt.title(f'Normalized {method} AvgRank by category \nANOVA={stat:.3f} pval={pval:.8f}')
		plt.xlabel(f'Normalized {method} AvgRank')
		plt.yticks([1,2], ['Solo', 'Multi'])	
		#plt.show()
		plt.close()

		print(f'=======================================')

if __name__ == '__main__':
	
	model_type = sys.argv[1]

	main(model_type, 'ideal')
	main(model_type, 'real')
	