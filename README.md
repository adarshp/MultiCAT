# MultiCAT

Code for the MultiCAT dataset paper (Findings of NAACL 2025).

The dataset is provided as a single SQLite3 database, and can be found at https://doi.org/10.5281/zenodo.14834836.

- The code for the score prediction results and the $\chi^2$ analyses are in the
  `score_prediction_and_chi2_analysis` folder.
- The code for the CLC detection baseline is in the `CLC` folder.
- The code for the DA/AP classification tasks as well as the LLaMA-3 baselines
  for sentiment/emotion classification is in the `DA` folder.
- The code for the sentiment and emotion recognition baselines is in the
  `sent_emo` folder.
- Additional documentation for the database will be added soon.

## License

The code and data are licensed under the Creative Commons CC-BY-4.0 license
(https://creativecommons.org/licenses/by/4.0/).

## Support

Please contact Adarsh Pyarelal (adarsh@arizona.edu) or open a Github issue on
this repository for any questions about the code or dataset.

## Contributing

Pull requests to improve the code or documentation are welcome.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{pyarelal-etal-2025-multicat,
    title = "{M}ulti{CAT}: Multimodal Communication Annotations for Teams",
    author = "Pyarelal, Adarsh  and
      Culnan, John M  and
      Qamar, Ayesha  and
      Krishnaswamy, Meghavarshini  and
      Wang, Yuwei  and
      Jeong, Cheonkam  and
      Chen, Chen  and
      Miah, Md Messal Monem  and
      Hormozi, Shahriar  and
      Tong, Jonathan  and
      Huang, Ruihong",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.61/",
    pages = "1077--1111",
    ISBN = "979-8-89176-195-7",
    abstract = "Successful teamwork requires team members to understand each other and communicate effectively, managing multiple linguistic and paralinguistic tasks at once. Because of the potential for interrelatedness of these tasks, it is important to have the ability to make multiple types of predictions on the same dataset. Here, we introduce Multimodal Communication Annotations for Teams (MultiCAT), a speech- and text-based dataset consisting of audio recordings, automated and hand-corrected transcriptions. MultiCAT builds upon data from teams working collaboratively to save victims in a simulated search and rescue mission, and consists of annotations and benchmark results for the following tasks: (1) dialog act classification, (2) adjacency pair detection, (3) sentiment and emotion recognition, (4) closed-loop communication detection, and (5) vocal (phonetic) entrainment detection. We also present exploratory analyses on the relationship between our annotations and team outcomes. We posit that additional work on these tasks and their intersection will further improve understanding of team communication and its relation to team performance. Code {\&} data: https://doi.org/10.5281/zenodo.14834835"
}
```
