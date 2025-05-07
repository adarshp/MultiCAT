# implementation of a basic random search
import os
from datetime import date
import shutil
import sys
from copy import deepcopy

sys.path.append("/home/u18/jmculnan/github/tomcat-dataset-creation")
import baselines.sent_emo.grid_config as config
from utils.baseline_utils import set_cuda_and_seeds


if __name__ == "__main__":
    if config.model_params.modality == "text":
        from baselines.sent_emo.text_only import text_only_train as train
    elif config.model_params.modality == "multimodal":
        from baselines.sent_emo.multimodal import multimodal_train as train
    else:
        exit(
            "This modality is not supported. Please set modality to 'text' or 'multimodal'."
        )

    # get device
    device = set_cuda_and_seeds(config)

    # get a copy of the params
    # this will not be altered during random search
    # the rest of the config file WILL be
    params = deepcopy(config.model_params)

    # load the dataset
    data = train.load_data(config)

    for m_type in params.model_type:
        # create save location
        output_path = os.path.join(
            config.exp_save_path,
            str(config.EXPERIMENT_ID)
            + "_"
            + m_type
            + "_"
            + config.EXPERIMENT_DESCRIPTION
            + str(date.today()),
        )

        print(f"OUTPUT PATH:\n{output_path}")

        # make sure the full save path exists; if not, create it
        os.system(f'if [ ! -d "{output_path}" ]; then mkdir -p {output_path}; fi')

        # copy the config file into the experiment directory
        shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

        all_model_info = []

        for fc_hid in params.fc_hidden_dim:
            for final_hid in params.final_hidden_dim:
                this_params = config.model_params
                this_params.fc_hidden_dim = fc_hid
                this_params.final_hidden_dim = final_hid
                this_params.model_type = m_type

                if len(all_model_info) > 0:
                    with open(
                        os.path.join(output_path, "grid_search_results.csv"), "a"
                    ) as f:
                        for item in all_model_info:
                            f.write(",".join(item))
                            f.write("\n")
                    all_model_info = []

                # add stdout to a log file
                with open(os.path.join(output_path, "log"), "a") as f:
                    if not config.DEBUG:
                        sys.stdout = f

                    sent_f1, emo_f1 = train.finetune(data, device, output_path, config)

                    all_model_info.append(
                        [
                            m_type,
                            str(params.lr),
                            str(fc_hid),
                            str(final_hid),
                            str(params.dropout),
                            str(round(sent_f1, 3)),
                            str(round(emo_f1, 3)),
                        ]
                    )
        # write the results of the final test in the search to file
        with open(os.path.join(output_path, "grid_search_results.csv"), "a") as f:
            for item in all_model_info:
                f.write(",".join(item))
