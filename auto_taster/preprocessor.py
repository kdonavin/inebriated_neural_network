import ipdb
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import spacy
from spacy.tokenizer import Tokenizer
from string import punctuation
import re

'''
Proprietary Values - NB Brewery asked that I do not share information contained
in this pickled dictionary object
'''
proprietary_vals = pkl.load(open('./proprietary_vals.pkl', 'rb'))

################################LOAD FEATURES################################

def load_data_batch_level(classified=True, hot_primary=True, reload=False):
	'''
	load data at the batch level.

	INPUT
		classified - boolean, if true: classify a single binary Hot value 
			based on beer tasting panel 'votes'. Else, y is returned as multi-
			dimensional Hot-or-Not panel 'votes'.
		hot_primary - boolean, whether to swap "Not" with Hot 
			classification values (i.e., "Not" = 1, "Hot" = 0). This simplifies
			prioritization of "Not" performance in model grid search.
		reload - boolean, whether to reload data from source, rather than from 
			pickle file

	OUTPUT
		X - DataFrame of features
		y - 1D array of labels
	'''
	if reload:

		#Sensory Data Aggregation
		sensory_df = clean_load_sensory()
		sensory_df = load_taster_quality(sensory_df)
		sensory_df = sensory_df.loc[sensory_df.is_validated == 1,] #NB 'validates' tasters for beer tasting skill 
		
		#Chemicals
		chem_df = clean_load_chem() 

		#Column selection 
		mc_cols = proprietary_vals['mc_cols']
		flavor_cols = column_range(sensory_df
			, proprietary_vals['flavor_col_start']
			, proprietary_vals['flavor_col_end'])
		chem_cols = column_range(chem_df
			, proprietary_vals['chem_col_start']
			, proprietary_vals['chem_col_end'])

		#Classified?
		if classified:
			labels = sensory_df.loc[:, ['id'] + mc_cols + flavor_cols + ['quality_group']].groupby('id').apply(hot_not_classifier)
			labels = pd.DataFrame(labels, columns = proprietary_vals['hot']) #for merging
			if hot_primary:
				labels = labels.replace(to_replace=[0,1], value=[1,0])
		else: 
			labels = sensory_df.loc[:, ['id'] + mc_cols + flavor_cols].groupby('id').mean()

		#Flavors
		flavors_df = sensory_df.loc[:, ['id']+ flavor_cols].groupby('id').max()

		#Join Data
		data = chem_df.join(flavors_df.loc[:, flavor_cols], how='inner')
		data = data.join(labels, how='inner') #DataFrame wrapper for classified "Hot or Not" Series object
		
		X_cols = chem_cols + flavor_cols

		if classified:
			y_cols = proprietary_vals['hot']
		else:
			y_cols = mc_cols

		X = data[X_cols]
		y = data[y_cols].values.flatten()

		pkl.dump((X, y), open('./data/nb_batch_level.pkl', 'wb'))

	else:

		X, y = pkl.load(open('data/nb_batch_level.pkl', 'rb'))

	return X, y

###############################SUPPORT METHODS###############################

def clean_chem(chem_data):
	'''
	Performs cleaning functions on New Belgium 
	chemical data
	'''

	chem_data['id'] = pd.Series([re.findall('[0-9]{9}', name)[0] if re.search('[0-9]{9}', name) else '' for name in chem_data.batch_name])
	chem_data.loc[:,:] = chem_data.loc[chem_data.id != '', :] #Remove test batches (7 last checked)
	chem_data.loc[:,'id'] = chem_data.site + chem_data.id
	chem_data.index = pd.Index(chem_data.id)

	#De-duplicate
	non_duplicates = chem_data.duplicated(subset=['id', 'component_name']) == False
	chem_data = chem_data.loc[non_duplicates, :]
	#Pivot chemicals into columns
	chem_data = chem_data.loc[:, ['id', 'component_name', 'result_value']].pivot(index='id', columns = 'component_name', values='result_value')
	chem_data.pop(np.nan) #this happens for some reason

	return chem_data

def clean_mc_cols(x):
	'''
	Cleaning multiple choice responses for Hot-or-Not columns
	'''
	if x == 2: #Not Hot
		return 0
	elif x == 0: #Hot - failed to answer
		return 1
	else: #Not sure, Hot
		return 1	

def clean_sensory(sensory_df, drop_empty_com=False, reclean_comments=False, \
		brand_dummies=True, code_dir = 'auto_taster'):
	'''
	Performs cleaning functions on fresh (from .csv) 
	sensory data

	INPUT
		sensory_df: sensory DataFrame to clean
		brand_dummies: whether to append dummy variable from NB brands
		code_dir: name of the directory where code is stored
	OUTPUT
	'''

	#Unique ID
	sensory_df.brew_number = sensory_df.brew_number.astype('str')
	sensory_df['id'] = sensory_df.site + sensory_df.brew_number

	#Taster name
	with open(code_dir + '/name_clean_dict.pkl', 'rb') as p:
		name_clean_dict = pkl.load(p)
	sensory_df.regname = [name.lower() for name in sensory_df.regname]
	sensory_df.regname = [name_clean_dict[name] if name in name_clean_dict.keys() else name for name in sensory_df.regname]

	#Brands
	sensory_df['flavor'] = pd.Series([re.sub('[{}]'.format(punctuation),'', flavor).lower() for flavor in sensory_df.flavor])
	if brand_dummies:
		sensory_df = sensory_df.join(pd.get_dummies(sensory_df.flavor))

	#Hot-or-Not MC
	MC_cols = proprietary_vals['mc_cols']
	for col in MC_cols: 
		sensory_df[col] = [clean_mc_cols(x) for x in sensory_df[col]]
	
	return sensory_df

def clean_load_chem():
	'''
	method to quickly load and clean the chemical data.
	'''
	chem = load_chem()
	chem = clean_chem(chem)
	return chem

def clean_load_sensory(subset=False, n=100):
	'''
	method to quickly load and clean a subset of the sensory 
	data.
	'''
	if subset:
		sensory = load_sensory_subset(n=n)
	else:
		sensory = load_sensory()
	sensory = clean_sensory(sensory)
	return sensory

def column_range(df, start_col, end_col):
	'''
	Helper method that returns a list of column names in df from start to finish

	INPUT
		df - DataFrame object
		start - starting column name
		finish - ending column name
	OUTPUT
		list of column names start to finish
	'''
	start = list(df.columns).index(start_col) #flavor dummies
	end = list(df.columns).index(end_col) #flavor dummies
	return list(df.columns[start:end+1])


def concatenate_sensory():
	'''
	Method to combine Fort Collins and Asheville brewery sensory
	data. Loads and saves a new .csv without input or output.
	'''
	
	sensory = pd.read_csv('data/sensory_ftc.csv', low_memory=False) #processed all at once rather than in chunks
	sensory2 = pd.read_csv('data/sensory_avl.csv', low_memory=False)
	sensory2.columns.values[4] = sensory.columns.values[4] #BrewNumber v Brew number	
	sensory['site'] = 'FTC'
	sensory2['site'] = 'AVL'
	sensory = pd.concat([sensory, sensory2])
	sensory.columns = process_column_names(sensory.columns)
	sensory.to_csv('data/sensory_clean.csv', index=False) #skip

def load_sensory():
	'''
	Load sensory data (FTC & AVL)
	'''
	return pd.read_csv('data/sensory_clean.csv', dtype={'brew_number': str})

def load_chem():
	'''
	Load chemical measurement data
	'''
	df_chem = pd.read_csv('data/chem.csv', dtype={'result_value': np.float64}, low_memory=False)

	#process column names
	df_chem.columns = process_column_names(df_chem.columns)
	return df_chem

def load_taster_quality(sensory_df, file='jan_code/name_group.pickle'):
	with open(file, 'rb') as p:
		group = pkl.load(p)
	group.columns = ['regname', 'quality_group']
	return pd.merge(sensory_df, group, on ='regname', how='left')

def process_column_names(col_index):
	'''
	process column names to be uniform: lower case, underscores

	INPUT: col_index Pandas Index object
	OUTPUT: replacement_col_index
	'''
	replacement_col_index = []
	for col_name in col_index:
		col_name = re.sub('([a-z])([A-Z])', r'\1 \2', col_name)
		replacement_col_index.append(col_name.lower().replace(' ','_'))
	return pd.Index(replacement_col_index)

def hot_not_classifier(sub_df, columns=proprietary_vals['mc_cols'], \
	quality_col='quality_group', threshold=0.9):
	'''
	Classify Hot or Not based on taster responses
	'''
	weighted_points = (sub_df.loc[:,columns].values * sub_df[quality_col].values.reshape(-1,1)).sum(axis=1)
	scale = (sub_df[quality_col]*5).sum() 
	p_ttb = pd.Series(weighted_points/scale).sum() #NOTE: ndarray will not sum with NaN values, must keep as pandas
	if p_ttb >= threshold:
		return 1
	else:
		return 0

if __name__ == '__main__': 
	pass

