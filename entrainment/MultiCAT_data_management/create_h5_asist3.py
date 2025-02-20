import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math
import h5py
from glob import glob
import random

def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup")
	parser.add_argument("--input_directory",
						default="./multicat_addressee_feats",
						help="directory for opensmile feat files (csv)")
	parser.add_argument("--h5_directory",
						default="./h5_output",
						help="directory for storing h5 files")
	return parser


def shuffle_files(input_dir):
	sessList = sorted(glob(input_dir + '/*.csv'))
	print("sessList: ", sessList)
	print("creating a list of shuffled feature files...")
	# print("sessList", sessList)
	random.shuffle(sessList)
	return sessList


#### Create Test Data file ####
def create_test(sessTest, h5_dir):
	spk_base = 1
	X_test = np.array([])
	for sess_file in sessTest:
		print("session file working on: ", sess_file)
		#load feature file as a dataframe
		df_i = pd.read_csv(sess_file, header=None)
		# load feature file as an array
		xx = np.array(df_i)
		print("feature array shape: ", xx.shape[0])
		N = xx.shape[0]
		if np.mod(N, 2) == 0:
			print("even number of data rows in feature array")
			print("N/2: ", math.floor(N / 2), N / 2)
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
		else:
			print("is not 0, taking modulus...")
			print("N/2: ", math.floor(N / 2))
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
			spk_label = np.append(spk_label, spk_base)
		xx = np.hstack((xx, spk_label.T.reshape([N, 1])))
		print(sess_file, "examined and test array created")
		X_test = np.vstack([X_test, xx]) if X_test.size else xx
		spk_base += 1
		# print(spk_base)
	# print(X_test[:10])

	X_test = X_test.astype('float64')
	hf = h5py.File(h5_dir + '/test_ASIST.h5', 'w')
	hf.create_dataset('dataset', data=X_test)
	hf.create_dataset('prosset', data=X_test[:, :24])
	hf.create_dataset('specset', data=X_test[:, 24:150])
	hf.create_dataset('vqset', data=X_test[:, 150:])
	hf.close()
	return None


def load_h5(file):
	print("loading h5 file: ", file)
	file = h5py.File(file, 'r')
	test = np.array(file['dataset'])
	print("loading complete!")
	return test


if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()

	input_directory = Path(args.input_directory).resolve()
	h5_output_directory = Path(args.h5_directory).resolve()
	print(f"input: {input_directory}\n output: {h5_output_directory}")
	
	#Check if output directory exists, if not, create it:   
	if not h5_output_directory.exists():
		h5_output_directory.mkdir(parents=True, exist_ok=True)
		print(f"Could not find specified output directory {h5_output_directory}. Creating directory...")
	else:
		print(f"Specified output directory {h5_output_directory} already exists. Continuing...")

	# Create h5 file
	sessTest = shuffle_files(args.input_directory)
	create_test(sessTest=sessTest, h5_dir=args.h5_directory)
	test_h5 = args.h5_directory + '/test_ASIST.h5'

	test_input = load_h5(test_h5)
	print(np.shape(test_input))
	print(test_input[:, 2])
