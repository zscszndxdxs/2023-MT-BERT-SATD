# ICSME-2023-MT-BERT-SATD
This repository is a replica package of the paper "Self-Admitted Technical Debts Identification: How Far Are We?", including the implementation code of MT-BERT-SATD, the preprocessed complete dataset used for training, and a tutorial on how to use our well-trained model for SATD identification across various sources.

## To avoid potential conflicts of interest, the original dataset collected in the article can be obtained from the following links
============================================================================
### dataset

| Dataset     | Sample Source | Link     |
| :---        |    :----   |          ---: |
| Dataset-01-Comments-Dockerfile | Code comments/dockerfile |[data](https://docs.google.com/spreadsheets/d/1ZCkdLxQjJyZpp88NtXYcSCNko8HX-2-uUzX217pf67s/edit#gid=0)  |
| Dataset-02-Comments-Python | Code comments/python |[data](https://github.com/DavidMOBrien/23Shades)  |
| Dataset-03-Comments-XML | Code comments/XML |[data](https://github.com/NAIST-SE/SATDinBuildSystems)  |
| Dataset-04-Comments-Java | Code comments/java |[data](https://zenodo.org/record/5825671)  |
| Dataset-05-Comments-Java | Code comments/java |[data](https://github.com/Naplues/MAT)  |
| Dataset-06-Comments-Java | Code comments/java |[data](https://github.com/ai-se/Jitterbug/tree/master/new_data/corrected)  |
| Dataset-07-Issue | Issue Trackers |[data](https://github.com/yikun-li/satd-issue-tracker-data)  |
| Dataset-08-Issue | Issue Trackers |[data](https://github.com/disa-lab/R-TD-SANER2022)  |
| Dataset-09-PR | Pull Requests |[data](https://zenodo.org/record/6829274)  |
| Dataset-10-PR | Pull Requests |[data](https://github.com/yikun-li/satd-different-sources-data)  |
| Dataset-11-Commits | Commit Messages |[data](https://github.com/yikun-li/satd-different-sources-data)  |

### code
The code implemented by the MT-BERT-SATD model can be found in the code file

### predict
1. Download the well-trained model [link](https://huggingface.co/aavvvv/mt-bert-satd/tree/main)
2. Put the downloaded three files "pytorch_model.bin", "vocab.txt" and "config.json" in the well_trained_model folder
3. 
