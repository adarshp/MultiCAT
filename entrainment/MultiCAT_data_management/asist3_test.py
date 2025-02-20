import os
############ Fix for issues with paths #######
## Get the current working directory
current_directory = os.getcwd()
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# Add the parent directory to the system path
sys.path.append(parent_dir)
############ 

import argparse
import numpy as np
import random
from feats.ecdc import *

def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup")
	parser.add_argument("--h5_directory",
						default="./asist3_h5",
						help="directory for storing h5 files")
	return parser


def load_h5(file):
	print("loading h5 file: ", file)
	file = h5py.File(file, 'r')
	test = np.array(file['dataset'])
	print("loading complete!")
	return test

def model_testing(model_name, X_test, cuda):
	model = VAE().double()
	model = torch.load(model_name)
	model.eval()
	if cuda:
		model.cuda()
	if 'l1' in model_name:
		p=1
	elif 'l2' in model_name:
		p=2
	else:
		print("need better model name")
		p=2

		pdb
	test_loss = 0
	fake_test_loss = 0
	Loss=[]
	Fake_loss = []
	# load h5 test file- iterating over conversations:
	for idx, data in enumerate(X_test):
		print("working on file: ", idx)
		print("length of test set list: ", len(data))

		x_data = data[:228]
		y_data = data[228:-1]

		# speaker- last item in list. Create a variable where the utterances
		# with the same speaker as the first utterance
		idx_same_spk =list(np.where(X_test[:,-1]==data[-1]))[0]

		# choose an item from the index same speaker which is not the same speaker
		ll = random.choice(list(set(idx_same_spk) - set([idx])))
		spk = int(data[-1])

		x_data = torch.from_numpy(x_data)
		y_data = torch.from_numpy(y_data)

		#Go back into list of lists, pull out randomly selected item,
		# and get the relevant cells from a different item of the same speaker
		#So, for an utterance by speaker A, replace it by a different utterance by speaker A
		# same for speaker B

		y_fake_data = X_test[ll,228:-1]

		y_fake_data = torch.from_numpy(y_fake_data)
		print("data loaded!")

		if cuda:
			x_data = x_data.cuda()
			y_data = y_data.cuda()
			y_fake_data = y_fake_data.cuda()

		recon_batch = model(x_data)

		z_x = model.embedding(x_data)
		z_y = model.embedding(y_data)
		# z_x = x_data
		# z_y = y_data
		loss_real = lp_distance(z_x, z_y, p).data
		# loss_real = loss_function(z_x, z_y, mu, logvar)

		#randomly selected fake item ? FIND OUT how the data is split
		# Is an item an utterance? A whole conversation?-
		# Take half the conversation, compare it to a real second half, and a fake second half?
		z_y_fake = model.embedding(y_fake_data)
		# z_y_fake = y_fake_data

		loss_fake = lp_distance(z_x, z_y_fake, p).data
		# loss_fake = loss_function(z_x, z_y_fake, mu, logvar)

		test_loss += loss_real
		fake_test_loss += loss_fake
		print("model run complete!")

	# this is inefficient- find a way to do everything on CPU
		Loss.append(loss_real.cpu())
		Fake_loss.append(loss_fake.cpu())

	# print loss_real, loss_fake
	test_loss /= X_test.shape[0]
	fake_test_loss /= X_test.shape[0]
	Loss = np.array(Loss)
	Fake_loss = np.array(Fake_loss)

	print("Total Real Loss: "+ str(test_loss))
	print("Total Fake Loss: " + str(fake_test_loss))

	print(float(np.sum(Loss < Fake_loss))/Loss.shape[0])
	return None

if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()

	# test_h5 = args.h5_directory + '/test_Fisher_nonorm.h5'
	test_h5 = args.h5_directory + '/test_ASIST.h5'
	test_input = load_h5(test_h5)
