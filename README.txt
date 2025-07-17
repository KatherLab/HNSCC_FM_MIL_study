# HNSCC_FM_MIL_study
Repository for data extraction and model extraction alongside model objects for the paper "End-to-end prediction of clinical outcomes in head and neck squamous cell carcinoma with foundation model-based multiple instance learning" by Meneghetti et al. 2025

The repository is organised into three sections:

*data_extraction: contains Python code for extraction of features for 2D, multiview and 3D modeling. Scripts output features into an .h5 format to use as inout for the modeling part of the STAMP package (https://www.nature.com/articles/s41596-024-01047-2). REQUIREMENTS.txt is only needed from STAMP==1.0.1., Once STAMP is installed, replace the files with the same name as the ones in STAMP_mods by the ones in the repository

*STAMP_mods: files to use with the STAMP 1.0.1 version release (https://github.com/KatherLab/STAMP/releases/tag/v1.0.1). Installation should not be very different than for the main version.

*Model objects: model objects and scripts for model loading for the different models presented in the manuscript both MIL and Radiomics.


Each subsection has their own README file to orient the reader.

*Preprocessing steps:
*Transform your .dcm files and tumor segmentation masks into NIFTIs in LAS orientation
*Resample into 1x1x1 mm voxels
*Create a body mask with TotalSegmentator
*Select a model and do feature extraction, wou can select the size of the crop there;  by default the body segmentation will be used to mask elements outside of it and the tumor segmentation masks will be used to select the CT slices containing tumor. The Centre of mass of the segmentation mask is then used as the center for a cropping which is then resized to 224x224 in 2D.
