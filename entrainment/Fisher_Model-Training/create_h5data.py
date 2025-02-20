# to run: use `train.py`
"""Code for splitting data files into training/validation/test splits and
computing L1 baseline."""

from os import path
from glob import glob
import h5py
import argparse
import random
import numpy as np
import pandas as pd
import math

# Major change: pd.read_csv(sess_file) to pd.read_csv(sess_file,header=None)
# this ensures that the top row of sessList was read as data, not header
# This significantly reduced the number of rows in the h5 file

def make_argument_parser():
	parser = argparse.ArgumentParser(
						description="Processing filepaths and values required for setup")
	parser.add_argument("--features_dir",
						default = "/home/tomcat/entrainment/feat_files/baseline_1_feats",
						help = "features directory")
	parser.add_argument("--h5_directory",
						default="/home/tomcat/entrainment/feat_files/baseline_1_h5",
						help= "directory for storing h5 files")
	return parser

""" 
Writing the shuffled list of feature files to a text file. This way, 
if you run into any issues while generating h5 files, 
the same randomized list of feature files is called, 
thus saving time and effort. Comment out 13-30 and uncomment 33-35 
if you wish to avoid saving the file list.
 Create h5 files 
"""
def split_files(feats_dir,sess_List= None):
	sess_files = path.isfile(sess_List)
	if sess_files == 1:
		print("list of transcripts exists")
		with open(sess_List, 'r') as f:
			temp = f.read().splitlines()
			# print(temp)
			# print(sorted(glob(feats_dir + '/*.csv')))
			print(len(temp), len(sorted(glob(feats_dir + '/*.csv'))))
			if len(temp) == len(sorted(glob(feats_dir + '/*.csv'))):
				print("list of shuffled files exists, importing...")
				sessList = temp
				print(sessList)
			else:
				print("error in importing files")
				raise ValueError("sessList.txt is not an accurate list of files")
	else:
		print("list of transcripts does not exist")
		sessList= sorted(glob(feats_dir + '/*.csv'))
		print("sessList: ", sessList)
		print("creating a list of shuffled feature files and saving to disk...")
		# print("sessList", sessList)
		random.seed(SEED)
		random.shuffle(sessList)
		with open(sess_List, 'w') as f:
			f.writelines( "%s\n" % i for i in sessList)

		with open(sess_List, 'r') as f:
			sessList = f.read().splitlines()

	#Alternative to 13-30:
	# sessList= sorted(glob.glob(feats_dir + '/*.csv'))
	# random.seed(SEED)
	# random.shuffle(sessList)

	num_files_all = len(sessList)
	num_files_train = int(np.ceil((frac_train*num_files_all)))
	print("num_files_train: ", num_files_train)
	num_files_val = int(np.ceil((frac_val*num_files_all)))
	print("num_files_val",num_files_val)
	num_files_test = num_files_all - num_files_train - num_files_val
	print("num_files_test",num_files_test)
	sessTrain = sessList[:num_files_train]
	sessVal = sessList[num_files_train:num_files_val+num_files_train]
	sessTest = sessList[num_files_val+num_files_train:]
	print(len(sessTrain) + len(sessVal) + len(sessTest))
	return(sessTrain, sessVal, sessTest)

#### Create Train Data file #####
def create_train(sessTrain, h5_dir):
	X_train =np.array([])
	X_train = np.empty(shape=(0, 0), dtype='float64' )
	for sess_file in sessTrain:
		print("working on file... ", sess_file)
		df_i = pd.read_csv(sess_file,header=None)
		xx = np.array(df_i)
		X_train = np.vstack([X_train, xx]) if X_train.size else xx
		print(sess_file, "examined and training array created")


	X_train = X_train.astype('float64')
	hf = h5py.File(h5_dir + '/train_Fisher_nonorm.h5', 'w')
	hf.create_dataset('dataset', data=X_train)
	hf.create_dataset('prosset', data=X_train[:,:24])
	hf.create_dataset('specset', data=X_train[:,24:150])
	hf.create_dataset('vqset', data=X_train[:,150:])
	hf.close()
	return None

#### Create Val Data file ####
def create_val(sessVal, h5_dir):
	if sessVal != []:
		print(sessVal, "exists and valid")

	X_val =np.array([])
	for sess_file in sessVal:
		print("working on file... ", sess_file)
		df_i = pd.read_csv(sess_file, header=None)
		xx = np.array(df_i)
		X_val = np.vstack([X_val, xx]) if X_val.size else xx
		print(sess_file, "examined and validation array created")
	print(X_val.shape)
	X_val = X_val.astype('float64')
	hf = h5py.File(h5_dir + '/val_Fisher_nonorm.h5', 'w')
	hf.create_dataset('dataset', data=X_val)
	hf.create_dataset('prosset', data=X_val[:,:24])
	hf.create_dataset('specset', data=X_val[:,24:150])
	hf.create_dataset('vqset', data=X_val[:,150:])
	hf.close()
	return None

#### Create Test Data file ####
def create_test(sessTest, h5_dir):
	spk_base = 1
	X_test =np.array([])
	for sess_file in sessTest:
		df_i = pd.read_csv(sess_file, header=None)
		xx = np.array(df_i)
		print("xx.shape[0]",xx.shape[0])
		N = xx.shape[0]
		if np.mod(N,2) == 0:
			print("is 0")
			print("N/2: ", math.floor(N / 2), N/2)
			spk_label = np.tile([spk_base, spk_base+1], [1, math.floor(N/2)])
		else:
			print("is not 0")
			print("N/2: ", math.floor(N / 2))
			spk_label = np.tile([spk_base, spk_base+1], [1, math.floor(N/2)])
			spk_label = np.append(spk_label, spk_base)
		xx = np.hstack((xx, spk_label.T.reshape([N,1])))
		print(sess_file, "examined and test array created")
		X_test=np.vstack([X_test, xx]) if X_test.size else xx
		spk_base += 1
		# print(X_test[:10])


	X_test = X_test.astype('float64')
	hf = h5py.File(h5_dir + '/test_Fisher_nonorm.h5', 'w')
	hf.create_dataset('dataset', data=X_test)
	hf.create_dataset('prosset', data=X_test[:,:24])
	hf.create_dataset('specset', data=X_test[:,24:150])
	hf.create_dataset('vqset', data=X_test[:,150:])
	hf.close()
	return None

if __name__ == "__main__":

	parser = make_argument_parser()
	args = parser.parse_args()
	SEED = 448
	frac_train = 0.8
	frac_val = 0.1

	tr, v, te = split_files(feats_dir = args.features_dir,
							sess_List=args.h5_directory+"/sessList.txt")
	create_train(sessTrain = tr, h5_dir = args.h5_directory)
	create_val(sessVal = v, h5_dir = args.h5_directory)
	create_test(sessTest = te, h5_dir = args.h5_directory)