# NCI-DOE-Collab-Pilot3-FrESCO-Framework-for-Exploring-Scalable-Computational-Oncology

## Description
The National Cancer Institute (NCI) monitors population level cancer trends as part of its Surveillance, Epidemiology, and End Results (SEER) program. This program consists of state or regional level cancer registries which collect, analyze, and annotate cancer pathology reports. From these annotated pathology reports, each individual registry aggregates cancer phenotype information and summary statistics about cancer prevalence to facilitate population level monitoring of cancer incidence. Extracting cancer phenotype from these reports is a labor intensive task, requiring specialized knowledge about the reports and cancer. Automating this information extraction process from cancer pathology reports has the potential to  improve not only the quality of the data by extracting information in a consistent manner across registries, but to improve the quality of patient outcomes by reducing the time to assimilate new data and enabling time-sensitive applications such as precision medicine. Here we present FrESCO: Framework for Exploring Scalable Computational Oncology, a modular deep-learning natural language processing (NLP) library for extracting pathology information from clinical text documents.

## User Community
Researchers interested in producing realistic synthetic clinical data.

## Usability	
To utilize FrESCO, users should have experience with Python and deep learning to set the model parameters appropriately. 
 
A GPU-powered computer would allow faster execution of FrESCO.

## Uniqueness	
FrESCO provides a modular artificial intelligence (AI) pipeline to take cancer pathology reports and make predictions about characteristics of the cancer. The individual modules making up the codebase allow for rapid prototyping of new model architectures, model profiling,and a compendium of the Modeling Outcomes using Surveillance data and Scalable Artificial Intelligence for Cancer ([MOSSAIC](https://datascience.cancer.gov/collaborations/nci-department-energy-collaborations/mossaic)) project to date.

## Components	

* Script: The [demo.py](./experiments/demo.py) file contains a demo script to train different methods on an example dataset, generate synthetic data, and produce performance metric reports.
* Data: The [UCIBreast.py](./datasets/UCIBreast.py) file contains code to prepare an example dataset from [UCI Machine Learning Repository Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer).

## Technical Details
Refer to this [README](./Technical_README.md).

## Reference
If you use FrESCO in your work please cite this repository. The bibtex entry is:
```
@misc{osti_1958817,
title = {FrESCO},
author = {Spannaus, Adam and Gounley, John and Hanson, Heidi and Chandra Shekar, Mayanka and Schaefferkoetter, Noah and Mohd-Yusof, Jamaludin and Fox, Zach and USDOE},
abstractNote = {The National Cancer Institute (NCI) monitors population level cancer trends as part of its Surveillance, Epidemiology, and End Results (SEER) program. This program consists of state or regional level cancer registries which collect, analyze, and annotate cancer pathology reports. From these annotated pathology reports, each individual registry aggregates cancer phenotype information and summary statistics about cancer prevalence to facilitate population level monitoring of cancer incidence. Extracting cancer phenotype from these reports is a labor intensive task, requiring specialized knowledge about the reports and cancer. Automating this information extraction process from cancer pathology reports has the potential to improve not only the quality of the data by extracting information in a consistent manner across registries, but to improve the quality of patient outcomes by reducing the time to assimilate new data and enabling time-sensitive applications such as precision medicine. Here we present FrESCO: Framework for Exploring Scalable Computational Oncology, a modular deep-learning natural language processing (NLP) library for extracting pathology information from clinical text documents.},
url = {https://www.osti.gov//servlets/purl/1958817},
doi = {10.11578/dc.20230227.2},
url = {https://www.osti.gov/biblio/1958817}, year = {2023},
month = {3},
note =
}
```

## Authors

 - Adam Spannaus (ORNL)
 - John P Gounley (ORNL)
 - Noah T Schaefferkoetter (ORNL)
 - Mayanka Chandra Shekar (ORNL)
 - Heidi Hanson (ORNL)
 - Zachary Fox (ORNL)
 - Jamaludin Mohd-Yusof (LANL)

LANL - Los Alamos National Laboratory; ORNL - Oak Ridge National Laboratory

## License

Oak Ridge National Laboratory released FrESCO under the MIT license.
 
For details, refer to [LICENSE](./LICENSE).
