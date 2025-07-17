Various goodies to preprocess CT images; includes:
-DICOM to NIFTI converter with parallel usage of CPU
-NIFTI resampler
-Small bash file example for invocation of Total Segmentator for body mask
-DICOM metadata extractors for CT; it assumes that every series is in a different folder.

*The typical pipeline would be:
*DICOM metadata extraction -> DICOMs to NIFTIs for masks and scans -> Resampler -> Feature extraction (in data_extraction directory of repo)
