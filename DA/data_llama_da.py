import os
import jsonlines

train_set_idx = [
    "T000602",
    "T000603",
    "T000604",
    "T000611",
    "T000612",
    "T000619",
    "T000620",
    "T000621",
    "T000622",
    "T000623",
    "T000624",
    "T000628",
    "T000631",
    "T000632",
    "T000635",
    "T000636",
    "T000637",
    "T000638",
    "T000713",
    "T000714",
    "T000715",
    "T000716",
    "T000719",
    "T000720",
    "T000723",
    "T000724",
    "T000729",
    "T000730",
]
valid_set_idx = [
    "T000614",
    "T000613",
    "T000607",
    "T000608",
    "T000633",
    "T000634",
]
test_set_idx = [
    "T000605",
    "T000606",
    "T000671",
    "T000672",
    "T000625",
    "T000626",
    "T000727",
    "T000728",
    "T000737",
    "T000738",
    "T000609",
    "T000610",
]

full_set_idx = train_set_idx + valid_set_idx + test_set_idx

train_path = "data/multicat/DA/train/"
test_path = "data/multicat/DA/test/"
valid_path = "data/multicat/DA/val/"


class Sample:
    def __init__(
        self,
        id,
        speaker,
        sentence,
        window_mask,
        num_spks,
        label,
        spk_names=None,
    ):
        self.id = id
        self.speaker = speaker
        self.sentence = sentence
        self.window_mask = window_mask
        self.num_spks = num_spks
        self.spk_names = spk_names
        self.label = label


def process_file(dirname, is_train):
    basic_label_map = {"S": 0, "B": 1, "D": 2, "F": 3, "Q": 4}
    basic_label_map = {
        "statement": "Statement",
        "backchannels": "Backchannel",
        "disruptions": "Disruption",
        "fillers": "Filler",
        "questions": "Question",
    }

    data = []
    count = 0
    fnames = os.listdir(dirname)
    j = 0
    for fname in fnames:
        if fname[-4:] != ".txt":
            continue
        count += 1
        f = open(os.path.join(dirname, fname), "r")

        spk_list = []
        utterance_list = []
        da_list = []
        spks_map = {}
        spk_ind = 0
        i = 0  # i tells curr utt number

        spk_names = []
        for l in f:
            speaker_id, text, DA = l.strip().split("|")
            spk_names.append(speaker_id)

            if speaker_id not in spks_map:
                spks_map[speaker_id] = spk_ind
                spk_ind += 1

            text = "<Speaker " + str(spks_map[speaker_id]) + ">:" + " " + text

            spk_list.append(spks_map[speaker_id])
            utterance_list.append(text)
            da_list.append(basic_label_map[DA])

            i += 1

        if not is_train:
            data.append(
                (
                    fname.split(".")[0],
                    (
                        spk_list,
                        utterance_list,
                        da_list,
                        len(spks_map),
                        spk_names,
                    ),
                )
            )
        else:
            data.append(
                (
                    fname.split(".")[0],
                    (
                        spk_list,
                        utterance_list,
                        da_list,
                        len(spks_map),
                        spk_names,
                    ),
                )
            )
            j += 1
    return data


def chunkify_dialogue(
    data, window_size: int = 0, chunk_size: int = 128, is_train=False
):
    total_chunks = 0.0
    total_chunk_spk = 0.0
    max_chunk_spk = 0.0
    split_data = []
    for (
        fname,
        dialogue,
    ) in data:  # split each dialogue into smaller chunks using a sliding window
        (
            spk_list,
            utterance_list,
            sent_list,
            emo_list,
            num_spks,
            spk_names,
        ) = dialogue
        len_dialogue = len(spk_list)  # full one meeting length
        i = 0
        for chunk_index in range(0, len_dialogue, chunk_size):
            total_chunks += 1
            spk_chunk = spk_list[
                max(0, chunk_index - window_size) : min(
                    chunk_index + chunk_size, len_dialogue
                )
            ]
            utterance_chunk = utterance_list[
                max(0, chunk_index - window_size) : min(
                    chunk_index + chunk_size, len_dialogue
                )
            ]
            spk_name_chunk = spk_names[
                max(0, chunk_index - window_size) : min(
                    chunk_index + chunk_size, len_dialogue
                )
            ]
            sent_chunk = sent_list[
                max(0, chunk_index - window_size) : min(
                    chunk_index + chunk_size, len_dialogue
                )
            ]
            emo_chunk = emo_list[
                max(0, chunk_index - window_size) : min(
                    chunk_index + chunk_size, len_dialogue
                )
            ]

            unique_spks = len(set(spk_chunk))
            total_chunk_spk += unique_spks
            if unique_spks > max_chunk_spk:
                max_chunk_spk = unique_spks
            window_mask = 0 if chunk_index == 0 or is_train else window_size
            chunk_key = fname + "_" + str(i)

            split_data.append(
                Sample(
                    chunk_key,
                    spk_chunk,
                    utterance_chunk,
                    window_mask,
                    num_spks,
                    emo_chunk,
                    sent_chunk,
                    spk_name_chunk,
                )
            )

            i += 1

    return split_data  # list of samples


train_data = process_file(train_path, True)
test_data = process_file(test_path, False)
valid_data = process_file(valid_path, False)

train_datas = chunkify_dialogue(train_data, chunk_size=19000, is_train=True)
test_datas = chunkify_dialogue(test_data, chunk_size=19000, is_train=False)
valid_datas = chunkify_dialogue(valid_data, chunk_size=19000, is_train=False)


instruction = (
    "You are an intelligent chatbot that can classify intentions for each speaker's utterance.\
                You will be given a list of possible dialogue act.\
                You will be given a sentence whose label needs to be predicted.\
                You will be given a snap shot of conversation from which you should predict the dialogue act label for given sentence.\
                Dialogy acts:\
                Statement\
                Backchannel\
                Disruption\
                Filler\
                Question\
                "
)


directory = "data/multicat/processed/DA/ctx5"

if not os.path.exists(directory):
    os.makedirs(directory)


train_data_write_path = os.path.join(directory, "train.json")
test_data_write_path = os.path.join(directory, "test.json")
val_data_write_path = os.path.join(directory, "val.json")


paths = [train_data_write_path, test_data_write_path, val_data_write_path]
datas = [train_datas, test_datas, valid_datas]


for path, data in zip(paths, datas):
    with jsonlines.open(path, mode="w") as writer:
        lines = []
        j = 0
        for my_sample in data:  # for each conv
            sentences = my_sample.sentence
            label_list = my_sample.label

            filename = my_sample.id.split("_")[0]
            assert len(sentences) == len(label_list)
            for i in range(len(sentences)):  # for each utt
                context = "Context: " + " ".join(
                    sentences[max(i - 5, 0) : i]
                    + [" # " + sentences[i] + " # "]
                    + sentences[i + 1 : i + 4]
                )

                line = {"instruction": None, "input": None, "output": None}
                sen = sentences[i]
                label = label_list[i]

                if label == "":
                    continue

                line["instruction"] = instruction
                line["input"] = context + " Sentence: " + sen
                line["output"] = label
                lines.append(line)

            j += 1
        writer.write(lines)
