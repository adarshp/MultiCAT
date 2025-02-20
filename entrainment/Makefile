# Makefile to automate entrainment detection analysis
# Change the values in the 'Parameters' block if you need to

# ----------
# Parameters
# ----------
# Location of the corpus (transcripts + sound files)
CORPUS=/media/mule/projects/ldc/fisher-corpus

# Output directory
OUTPUT_DIR=feat_files

# Location of the corpus sound files 
AUDIO_DIR_ROOT=$(CORPUS)/fisher_eng_tr_sp_LDC2004S13_zip

# Path to OpenSMILE configuration files
OPENSMILE_CONFIG_BASELINE =scripts_and_config/emobase2010_haoqi_revised.conf
OPENSMILE_CONFIG_NED=scripts_and_config//emobase2010_mod.conf

# Directory with all transcripts
TRANSCRIPT_DIR=$(CORPUS)/fe_03_p1_tran/data/trans/all_trans

# -------------------------------------------------------------------

# Find all .sph files recursively inside AUDIO_DIR_ROOT
SPH_FILES:= $(shell find $(AUDIO_DIR_ROOT) -type f -name '*.sph')

# Replace .sph suffix with .wav suffix, then replace AUDIO_DIR_ROOT with feat_files
# (output files will go into the feat_files folder).
WAV_FILES:= $(patsubst $(AUDIO_DIR_ROOT)%, feat_files%, $(SPH_FILES:.sph=.wav))

# Construct file names for baseline raw CSV files
BASELINE_RAW_CSV_FILES:= $(WAV_FILES:.wav=_features_raw_baseline.csv)

# Construct file names for NED (neighborhood entrainment distance) raw CSV files
NED_RAW_CSV_FILES:= $(WAV_FILES:.wav=_features_raw_ned.csv)

# Construct file names for baseline normed CSV files. 
BASELINE_NORMED_CSV_FILES:= $(WAV_FILES:.wav=_features_normed_baseline.csv)

# Construct file names for NED normed CSV files. 
# NED_NORMED_CSV_FILES:= $(WAV_FILES:.wav=_features_normed_ned.csv)

# all: $(NED_NORMED_CSV_FILES) $(BASELINE_NORMED_CSV_FILES)
all: $(BASELINE_NORMED_CSV_FILES)

# Recipe to convert .sph files to .wav files
$(OUTPUT_DIR)/%.wav: scripts/sph2wav $(AUDIO_DIR_ROOT)/%.sph
	@mkdir -p $(@D)
	$^ $@

$(OUTPUT_DIR)/%_features_raw_baseline.csv: $(OUTPUT_DIR)/%.wav
	SMILExtract -C $(OPENSMILE_CONFIG_BASELINE) -I $< -O $@

#$(OUTPUT_DIR)/%_features_raw_ned.csv: $(OUTPUT_DIR)/%.wav
#	SMILExtract -C $(OPENSMILE_CONFIG_NED) -I $< -O $@

# We define the special target .SECONDEXPANSION in order to handle expansions
# in prerequisites ($$).
.SECONDEXPANSION:
$(OUTPUT_DIR)/%_features_normed_baseline.csv: feats/feat_extract_nopre.py\
							 $(OUTPUT_DIR)/%_features_raw_baseline.csv\
							 $(TRANSCRIPT_DIR)/$$(notdir %).txt
	$^ $@

# We define the special target .SECONDEXPANSION in order to handle expansions
# in prerequisites ($$).

# .SECONDEXPANSION:
#$(OUTPUT_DIR)/%_features_normed_ned.csv: feats/feat_extract_nopre.py\
#							 $(OUTPUT_DIR)/%_features_raw_ned.csv\
#							 $(TRANSCRIPT_DIR)/$$(notdir %).txt
#	$^ $@

webapp:
	PYTHONPATH="." streamlit run webapp/main.py --server.port=8085 --server.address=localhost
