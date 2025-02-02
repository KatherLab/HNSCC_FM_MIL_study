# HNSCC_FM_MIL_study
Repository for data extraction and model extraction alongside model objects for the paper "End-to-end prediction of clinical outcomes in head and neck squamous cell carcinoma with foundation model-based multiple instance learning" by Meneghetti et al. 2025

The repository is organised into three sections:

*data_extraction: contains Python code for extraction of features for 2D, multiview and 3D modeling. Scripts output features into an .h5 format to use as inout for the modeling part of the STAMP package (https://www.nature.com/articles/s41596-024-01047-2). REQUIREMENTS.txt is only needed from STAMP==1.0.1 

*STAMP_mods: files to use with the STAMP 1.0.1 version release (https://github.com/KatherLab/STAMP/releases/tag/v1.0.1). Installation should not be very different than for the main version.

*Model objects: model objects and scripts for model loading for the different models presented in the manuscript both MIL and Radiomics.


Each subsection has their own README file to orient the reader.
