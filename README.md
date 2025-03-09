# MultiCAT

Code and data for the MultiCAT dataset paper (Findings of NAACL 2025).

## Paper

The paper that introduces the dataset is included in this repository
(`multicat_naacl_2025_findings.pdf`). A [data
statement](https://aclanthology.org/Q18-1041/) is included in the appendices.

## Dataset

The dataset is provided as a single SQLite3 database
(`datasette_interface/multicat.db`).  The dataset has also been uploaded to
Zenodo to associate it with a DOI, as well as to ensure long-term data
preservation https://doi.org/10.5281/zenodo.14834836.

## Baselines

- Baselines
  - The code for the CLC detection baseline is in the `CLC` folder.
  - The code for the DA/AP classification tasks as well as the LLaMA-3 baselines
    for sentiment/emotion classification is in the `DA` folder.
  - The code for the other baseline results will be added soon.
- The code for the score prediction results and the $\chi^2$ analyses are in the
  folder `score_prediction_and_chi2_analysis`.


## Web UI

The MultiCAT dataset can also be explored interactively via a web browser at
https://multicat.lab.pyarelal.xyz. If you want to run the UI locally, please see
`datasette_interface/README.md`.

## License

The code and data are licensed under the Creative Commons CC-BY-4.0 license
(https://creativecommons.org/licenses/by/4.0/).

## Support

Please contact Adarsh Pyarelal (adarsh@arizona.edu) or open a Github issue on
this repository for any questions about the code or dataset.

## Contributing

Pull requests to improve the code or documentation are welcome.
