#!/usr/bin/env python

import os
import csv
import pandas as pd
import numpy as np
import argparse
from subprocess import run, check_output

# ------------------------------------------------------------------------
# Params Setup
# ------------------------------------------------------------------------


def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup"
									)
	parser.add_argument("raw_features_csv", help="Input raw features CSV")
	parser.add_argument("transcript", help="Input transcript")
	parser.add_argument("output_csv", help="output_normed_features_csv")
	parser.add_argument("--norm", type=bool, required=False, default=True,
						help="do session level normalization or not",
						)
	parser.add_argument("--window_size", required=False, type=float, default=10)
	parser.add_argument("--shift_size", required=False, type=float, default=1)
	parser.add_argument("--writing", required=False, type=str, default=True,
						help="whether raw features need to be saved or not.",
						)
	parser.add_argument("--IPU_gap", required=False, type=float, default=50)
	return parser


def final_feat_calculate(sample_index, all_raw_norm_feat, all_raw_feat_dim):
	whole_output_feat = np.array([], dtype=np.float32).reshape(
		0, all_raw_feat_dim * 6
	)
	for idx_frame in sample_index:
		tmp_all_raw_norm_feat = np.copy(all_raw_norm_feat[idx_frame, :])
		funcs_per_frame = func_calculate(tmp_all_raw_norm_feat)
		whole_output_feat = np.concatenate(
			(whole_output_feat, funcs_per_frame), axis=0
		)
	return whole_output_feat


def final_feat_calculate_multicat(row_ends, all_raw_norm_feat, all_raw_feat_dim):
	whole_output_feat = np.array([], dtype=np.float32).reshape(
		0, all_raw_feat_dim * 6
	)
	# if we are trying to get func_calculate to run over an entire utterance
	# and only one utterance, utt_feats is this array
	tmp_all_raw_norm_feat = np.copy(all_raw_norm_feat[row_ends[0]:row_ends[1], :])
	# tmp_all_raw_norm_feat = np.copy(utt_feats)
	print(f"Shape of all_raw_norm_feat: {np.shape(tmp_all_raw_norm_feat)}")
	funcs_per_frame = func_calculate(tmp_all_raw_norm_feat)
	print(f"Shape of funcs_per_frame: {np.shape(funcs_per_frame)}")
	whole_output_feat = np.concatenate(
		(whole_output_feat, funcs_per_frame), axis=0
	)
	return whole_output_feat


def func_calculate(input_feat_matrix):
	"""
	Given a numpy array calculate its statistic functions
	6 functions: mean, median, std, perc1, perc99, range99-1
	"""
	output_feat = np.array([], dtype=np.float32).reshape(1, -1)
	num_feat = input_feat_matrix.shape[1]
	for i in range(num_feat):
		tmp = input_feat_matrix[:, i]
		tmp_no_nan = tmp[~np.isnan(tmp)]
		if tmp_no_nan.size == 0:
			mean_tmp = 0
			std_tmp = 0
			median_tmp = 0
			perc1 = 0
			perc99 = 0
			range99_1 = 0
		else:
			mean_tmp = np.nanmean(tmp)
			std_tmp = np.nanstd(tmp)
			median_tmp = np.median(tmp_no_nan)
			tmp_no_nan_sorted = np.sort(tmp_no_nan)
			total_len = tmp_no_nan_sorted.shape[0]
			perc1_idx = np.int_(np.ceil(total_len * 0.01))
			if perc1_idx >= total_len:
				perc1_idx = 0
			perc99_idx = np.int_(np.floor(total_len * 0.99))
			if perc99_idx < 0 or perc99_idx >= total_len:
				perc99_idx = total_len - 1
			perc1 = tmp_no_nan_sorted[perc1_idx]
			perc99 = tmp_no_nan_sorted[perc99_idx]
			range99_1 = perc99 - perc1
		# append for one
		new_func = np.array(
			[mean_tmp, median_tmp, std_tmp, perc1, perc99, range99_1]
		)
		new_func = np.reshape(new_func, (1, 6))
		output_feat = np.hstack((output_feat, new_func))

	return output_feat


def create_normed_features_csv(
	raw_features_csv,
	transcript,
	window_size,
	shift_size,
	norm,
	IPU_gap,
	writing,
):

	# ------------------------------------------------------------------------
	# load transcript timings
	# ------------------------------------------------------------------------
	## TODO: load the file from trans/, store it in an array in start, end, spk (A or B) fmt

	spk_list = []

	with open(transcript) as f:
		trans = f.readlines()

	for line in trans:
		if line != "\n":
			if line[0] != "#":
				start, stop, spk = line.split(":")[0].split(" ")
				spk_list.append([start, stop, spk])

	## ------------------------------------------------------------------------
	## functional calculation: this has to be  PER Utterance (for entrainment)
	## ------------------------------------------------------------------------
	## frame length and overlap size in seconds
	# frame_len = window_size/0.01
	# frame_shift_len = shift_size/0.01

	# read csv feature file
	csv_feat = pd.read_csv(
		raw_features_csv, dtype=np.float32, on_bad_lines="warn"
	)
	csv_feat = csv_feat.values.copy()
	print("feature array has the following shape: ", np.shape(csv_feat))
	feat_data = np.copy(csv_feat)
	# convert the first column index to int index
	sample_index = list(map(int, list((feat_data[:, 0]))))

	## Orginal code line 178
	# def turn_level_index(spk_list, sample_index):
	# '''generate indices for different turns'''

	turn_level_index_list = []
	last_spk = "A"
	s2_found = True
	gap_found = True
	for spch in spk_list:
		start = int(float(spch[0]) / 0.01)
		stop = int(float(spch[1]) / 0.01)
		spk = spch[2]
		if not turn_level_index_list:
			turn_level_index_list = [sample_index[start:stop]]
			last_stop = stop
			continue
		if spk == last_spk:
			if start - last_stop < IPU_gap / 10:
				turn_level_index_list[-1].extend(sample_index[start:stop])

			else:
				if s2_found:
					if gap_found:
						turn_level_index_list[-1] = sample_index[start:stop]
					else:
						turn_level_index_list.append(sample_index[start:stop])

				else:
					turn_level_index_list.append(sample_index[start:stop])
					s2_found = True
				gap_found = True
		else:
			if not gap_found:
				turn_level_index_list.append(turn_level_index_list[-1])
			gap_found = False
			s2_found = False
			turn_level_index_list.append(sample_index[start:stop])
		last_stop = stop
		last_spk = spk

	if len(turn_level_index_list) % 2 == 1:
		turn_level_index_list = turn_level_index_list[:-1]

	s1_list = []
	s2_list = []
	for i, itm in enumerate(turn_level_index_list):
		if i % 2 == 0:
			s1_list.append(itm)
		else:
			s2_list.append(itm)

	"""
	## feature selection and normalization (original comments)

	- remove the mean for mfcc
	- normalize for pitch = log(f_0/u_0)
	- normalize for loudness
	"""
	# f0 normalization
	f0 = np.copy(feat_data[:, 70])

	# replace 0 in f0 with nan, get mean
	f0[f0 == 0.0] = np.nan
	f0_mean = np.nanmean(f0)

	if norm:
		f0[~np.isnan(f0)] = np.log2(f0[~np.isnan(f0)] / f0_mean)

	f0 = np.reshape(f0, (-1, 1))
	# f0_de normalization
	f0_de = np.copy(feat_data[:, 74])
	f0_de[f0_de == 0.0] = np.nan

	if norm:
		f0_de_mean = np.nanmean(np.absolute(f0_de))
		f0_de[~np.isnan(f0_de)] = np.log2(
			np.absolute(f0_de[~np.isnan(f0_de)] / f0_de_mean)
		)

	f0_de = np.reshape(f0_de, (-1, 1))

	# intensity normalization
	intensity = np.copy(feat_data[:, 2])
	if norm:
		int_mean = np.mean(intensity)
		intensity = intensity / int_mean

	intensity = np.reshape(intensity, (-1, 1))
	# intensity_de normalization
	intensity_de = np.copy(feat_data[:, 36])

	if norm:
		int_de_mean = np.mean(intensity_de)
		intensity_de = intensity_de / int_de_mean

	intensity_de = np.reshape(intensity_de, (-1, 1))
	# feat_idx = range(3,34) + range(37, 68)   with spectral de
	feat_idx = list(range(3, 34))
	mfcc_etc = np.copy(feat_data[:, feat_idx])
	if norm:
		mfcc_etc_mean = np.mean(mfcc_etc, axis=0)
		mfcc_etc_mean.reshape(-1, 1)
		mfcc_etc_norm = mfcc_etc - mfcc_etc_mean
	else:
		mfcc_etc_norm = np.copy(mfcc_etc)

	# jitter and shimmer normalization
	idx_jitter_shimmer = [71, 72, 73]
	jitter_shimmer = np.copy(feat_data[:, idx_jitter_shimmer])
	jitter_shimmer[jitter_shimmer == 0.0] = np.nan

	if norm:
		jitter_shimmer_mean = np.nanmean(jitter_shimmer, axis=0)
		jitter_shimmer_mean.reshape(-1, 1)
		jitter_shimmer_norm = jitter_shimmer - jitter_shimmer_mean
	else:
		jitter_shimmer_norm = jitter_shimmer

	##-----------------------------------------------------------------------
	## function calculation
	##-----------------------------------------------------------------------
	all_raw_norm_feat = np.hstack(
		(
			f0,
			f0_de,
			intensity,
			intensity_de,
			jitter_shimmer_norm,
			mfcc_etc_norm,
		)
	)

	# feature dimension
	all_raw_feat_dim = all_raw_norm_feat.shape[1]

	whole_func_feat1 = final_feat_calculate(
		s1_list, all_raw_norm_feat, all_raw_feat_dim
	)
	whole_func_feat2 = final_feat_calculate(
		s2_list, all_raw_norm_feat, all_raw_feat_dim
	)
	whole_func_feat = np.hstack((whole_func_feat1, whole_func_feat2))

	##-----------------------------------------------------------------------
	## normalization at whole session level, using scikit-learn
	## -- for each feature 0 mean and 1 variance
	##-----------------------------------------------------------------------
	# write to csv file

	if writing == True:
		print(f"Writing IPU-level features to file {output_csv}")
		feat_csv_file_name = output_csv
		with open(
			output_csv, "w"
		) as fcsv:  # changed 'wb' to 'w' to avoid TypeError
			writer = csv.writer(fcsv)
			writer.writerows(whole_func_feat)
		print("file ", feat_csv_file_name, " processed!")


if __name__ == "__main__":

	parser = make_argument_parser()
	args = parser.parse_args()

	create_normed_features_csv(
		args.raw_features_csv,
		args.transcript,
		args.window_size,
		args.shift_size,
		args.norm,
		args.IPU_gap,
		args.writing,
	)

