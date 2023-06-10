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
3. Place the unclassified CSV file, such as "4_unclassified.csv", in the unclassified_files directory, and execute the following code to perform the identification of Self-Admitted Technical Debt (SATD).
```
python predict.py --task {id 1-5 } --data_dir {file_name} --output_dir {out_path}.
```
#### illustrate: \<br> 
The optional range of {task} is 1-5, where they represent unclassified files from 1- "Issue Trackers", 2- "Pull Requests", 3- "Commit Messages", 4- "Code Comments", and 5- "Others".\<br> 


For example, "4_unclassified" in the example represents data in code comments, so the detailed running code is:
```
python predict.py --task 4 --data_dir 4_unclassified --output_dir predict_files.
```
#### The prediction results will be output in the {output_dir} folder.
