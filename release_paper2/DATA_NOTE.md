# Data note

This package uses the Open University Learning Analytics Dataset (OULAD):
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

OULAD is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).
The raw data are NOT included in this package.

How to obtain: download the dataset from the Open University Learning Analytics Dataset page
(https://analyse.kmi.open.ac.uk/open_dataset) or from the UCI Machine Learning Repository
("Open University Learning Analytics dataset"), and place the seven CSV files (assessments.csv, courses.csv,
studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv, vle.csv) in
`data/raw/oulad/`. The scripts read them from there (mounted at /workspace/data inside the container).
