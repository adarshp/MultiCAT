import os
import csv
import pandas as pd
import argparse
from pathlib import Path
import re

def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup")
	parser.add_argument("--input_directory",
						default="./entrainment_annotations",
						help="directory for providing input files")
	parser.add_argument("--output_directory",
						default="./files_for_dyad_generation",
						help="directory for storing output dyad files")
	return parser

def open_files(dir, extension : str = "tsv"):
	file_list = []
	for file in os.listdir(dir):
		ext = file.split('.')
		if ext[-1] == extension:
			file_list.append(file)
	return file_list

def extract_utterances(input_dir, output_dir, file_ls, unique_trials=None):
	"""
	This function searches through a list of speaker-separated transcripts
	in a given directory, and lists the unique trial IDs it finds in the file names.
	Then, it loops over all the transcripts to append turns from each
	speaker-separated transcript file into a consolidated dataframe,
	corresponding to their trial IDs, removes NAN, sorts by start time,
	and saves each as a csv file without the speaker IDs.

	User can also provide a list of Trial IDs to run function only on them.
	"""

	if not unique_trials:
		print("No user-provided list of trials. Processing all files in directory...")
	files = []
	trials = []

	for file in file_ls:
		## Extract filename without extension:
		filename = file.split(".")[0]
		files.append(filename)
		## Get trial ID:
		trial = re.search(r"-(T[a-zA-Z0-9]+)_", filename).group(1)
		if not unique_trials:
			trials.append(trial)

		elif unique_trials and trial in unique_trials:
			print(f"File found for trial: {trial}")
			trials = unique_trials
	
	print("Trials being processed: ", list(set(trials)))
	
	count = 0
	
	# loop over trials found in list:
	for unique_trial in list(set(trials)):
		print(f"searching for transcripts for trial: {unique_trial}...")
		filesavename = None
		i_df = None

		## loop over files in input directory to search for those corresponding
		## to trial of interest
		for filename in files:
			trial = re.search(r"-(T[a-zA-Z0-9]+)_", filename).group(1)
			## print(f"processing files of trial: {trial}")

			## if current file corresponds to trial of interest:
			if trial == unique_trial:
				df = pd.read_csv(input_dir+"/"+filename+".tsv", sep='\t', encoding='utf8')
				## print(f"trial information for {trial} found, processing transcript: ", filename)
				
				## remove empty lines:
				df = df.dropna(how="all")
				print(f"turns found in {filename} is {df.shape[0]}")

				## if combined transcript dataframe doesn't exist, create it,
				## construct its filename, and populate it will all turns in current dataframe
				##  Else, append all turns to combined dataframe for current trial
				if i_df is None:
					print("i_df does not exist")
					filename_without_member = re.sub(r"-E[a-zA-Z0-9]+_", "-NA_", filename)
					filesavename = output_dir + "/" + filename_without_member + ".csv"
					i_df = df
				else:
					print("i_df exists")
					i_df = pd.concat([i_df, df], axis = 0)
	
			## if current file doesn't correspond to the trial of interest, move on:
			else:
				print("Continuing search...")
			
		## remove rows with all empty columns
		i_df.dropna(how="all", axis=1, inplace=True)

		## remove colon in addressee columns		
		i_df["addressee"] = i_df["addressee"].apply(lambda x: x.split(":")[0])

		## sort column by start time
		i_df= i_df.sort_values(by = ["start"])		

		## save csv file
		print(f"turns found in {filesavename} is {i_df.shape[0]}")
		i_df.to_csv(filesavename, index = False, sep = ",")
		count += 1
		print(f"file {filesavename} saved to output folder!")
		# print("trial dataframe:")
		# print("rows: ", i_df.axes[0])
		# print("cols: ", i_df.axes[1])
	print(f"Search completed! Output folder should have {count} files.")
	return None

if __name__ == "__main__":
	## Read arguments:
	parser = make_argument_parser()
	args = parser.parse_args()

	input_dir = args.input_directory
	output_dir = args.output_directory
	print(f"input directory: {input_dir}\n output directory: {output_dir}")

	#Check if output directory exists, if not, create it:   
	if not Path(output_dir).exists():
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		print(f"Could not find specified output directory {output_dir}. Creating directory...")
	else:
		print(f"Specified output directory {output_dir} already exists. Continuing...")
	files = open_files(input_dir)
	# print(f"files: {files}")

	## uncomment the following 2 line 
	## to only read a subset of trials:
	# unique_trials = ["T000723"]
	# x = extract_utterances(input_dir, output_dir, files, unique_trials)

	## Clean up transcript and consolidate turns for each trial:
	x = extract_utterances(input_dir, output_dir, files)