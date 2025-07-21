rom dcmrtstruct2nii import dcmrtstruct2nii
import dicom2nifti
import os
import pandas as pd
from pathlib import Path
import multiprocessing.dummy as mp
import pydicom as pyd
import pydicom_seg as pyd_seg
import SimpleITK as sitk
import argparse
################################################################################
#Parser
################################################################################
parser = argparse.ArgumentParser(description="Inputs for the DICOM 2 Nift ti functions. Make sure all images and segmentations follow LAS orientation!")
parser.add_argument('--pth_csv', type=str, required=True, help="Path to where the images and segmentations are, each folder in the path should be one patient")
parser.add_argument('--pth_out', type=str, required=True, help = "Output directory where you want the features")
parser.add_argument("--type_im", type=str, default="DICOM", required=False, help="Type of files that you would like to process; options are DICOM, SEG and RTSTRUCT, default: DICOM")
parser.add_argument("--n_workers", type=int, default=20, required=False, help="Input the number of CPUs that will be used, default:20")
args= parser.parse_args()

pth= Path(args.pth_csv)
outdir= Path(args.pth_out)
type_im= args.type_im
n_works= args.n_workers
if type_im =="RTSTRUCT":
    print("This will transform the DICOM and RTSTRUCT structures, but will keep their orientations; please check up the orientatiosn afterwards and make sure to change them to LAS through either MONAI or NIBABEL")
################################################################################
#Function definition
################################################################################
def DICOM_2_nii_RTStruct(list_elemnt, foreground: int=1, background: int=0):
    try:
        pat_list,path_DICOM, path_RTstruct, outdir = list_elemnt
        os.makedirs(os.path.join(outdir,pat_list), exist_ok=True)
        set_dir = set(os.listdir(os.path.join(outdir,pat_list)))
        
        if len(set_dir) !=0:
            print("Patient already transformed:" + pat_list+", or check directory, it is non-empty!")
        else:
            print("Transforming patient " + pat_list)
            dcmrtstruct2nii(rtstruct_file=path_RTstruct,
                            dicom_file=path_DICOM,
                            mask_foreground_value= foreground,
                            mask_background_value= background,
                            output_path= os.path.join(outdir,pat_list)) 
            
    except: 
        print("Something wrong happened with patient " + pat_list)


def DICOM_2_nii_SEG(list_elmn, name_out:str="SEG_structure"):
    """
    Transform DICOMS with SEG masks into NIFTI files
    """
    pat_list,path_DICOM, path_SEG, outdir = list_elmn
    print("Transforming patient " + pat_list)
    os.makedirs(os.path.join(outdir,pat_list), exist_ok=True)
    try:
        dicom2nifti.dicom_series_to_nifti(original_dicom_directory=path_DICOM,output_file=os.path.join(outdir,pat_list,"image.nii.gz"), reorient_nifti=True) 
    except: 
        print("something happened with conversion of the main CT image for patient " + pat_list)
    #This reorients data to be LAS-defined as in nibabel, we have to keep it in mind for the mask!
    try:
        segmentation = pyd.dcmread(fp = os.path.dirname(path_SEG))
        reader = pyd_seg.SegmentReader()
        segmentation = reader.read(segmentation)
        num = 1
        for segment_number in segmentation.available_segments:
            image = segmentation.segment_image(segment_number)
            image = sitk.DICOMOrient(image, 'LAS') #reorient to the same way the image is from dicom2nii
            if len(segmentation.available_segments)==1:
                nm= name_out +".nii.gz"
                sitk.WriteImage(image,os.path.join(outdir,pat_list,nm))     
              #Get it to SITK image
            else :
                num+=1
                nm = name_out + str(num) + ".nii.gz"
                sitk.WriteImage(image,os.path.join(outdir,pat_list,nm))
        del image
        del segmentation
    except:
        print("Something wrong happened with patient " + pat_list)  

def DICOM_2_nii_DCM(list_elmn, name_out:str="main_CT.nii.gz"):
    pat_list,path_DICOM, path_SEG, outdir = list_elmn
    print("Transforming patient " + pat_list)
    os.makedirs(os.path.join(outdir,pat_list), exist_ok=True)
    try:
        dicom2nifti.dicom_series_to_nifti(original_dicom_directory=path_DICOM,output_file=os.path.join(outdir,pat_list,name_out), reorient_nifti=True) #This makes the NIFTI LAS oriented
    except: 
        print("Something happened with conversion of the main CT image for patient " + pat_list)
    


def parallel_DICOM_2_nii(type: str, dicom_df_path: Path, outdir: Path, n_workers: int=20):
    '''
    Parallel function for DICOM to NIFTI conversion with segmentation files.

    Inputs:

    type: str that should be one of RTSTRUCT, SEG or DICOM, DICOM processes only DICOM files, SEG creates from DICOM-SEG files that have segmentation data within the main DICOM and RTSTRUCT processes RTSTRUCT files.
    
    dicom_df_path: Path to the csv containing the information about DICOM files folders and their segmentations. Needs 3 columns: PATIENT, DICOM_Path and Segmentation_path. Both paths should be absolute and point to the folder, not any specific file within.
    For patients with more than 1 scan, it is recommended to name them differently in the csv, otherwise they will be skipped.
    
    n_workers: int, number of CPU cores that will transform each patient, one patient per core.
    
    outdir: Path, absolute path to where you want the transformed files to be saved in. One subfolder per each PATIENT will be created there.

    Outputs: 
    A new directory in outdir with the transformed images + segmentations
    '''

    df=pd.read_csv(dicom_df_path)
    outs= [outdir]*len(df)
    outs=pd.Series(outs)
    try:
        assert type in ["RTSTRUCT","SEG","DICOM"]
    except:
        AssertionError("Type a type of structure file for the Dicom which is either RTSTRUCT or SEG")
    
    inpt = list(zip(df["PATIENT"], df["DICOM_path"],df["Segmentation_path"],outs))
    if type =="RTSTRUCT":
        with mp.Pool(n_workers) as pl:
            
            print("The transformation will output all structures in the RTSTRUCT file, backgrpund of the segmentation will be 0 and the mask will be 1")
            print("This function transforms the input into NIFTIs, but does not guarantee LAS orientation; please check your NIFTIs afterwards and convert them with MONAI/Nibabel")
            try:
                pl.map(DICOM_2_nii_RTStruct,inpt) 
            except KeyboardInterrupt:
                print("Interrupted the conversion. Terminating workers")

    if type== "SEG":
        with mp.Pool(n_workers) as pl:
            try:
                pl.map(DICOM_2_nii_SEG,inpt)
            except KeyboardInterrupt:
                print("Interrupted the conversion. Terminating workers")
    if type=="DICOM":
            try:
                pl.map(DICOM_2_nii_DCM,inpt)
            except KeyboardInterrupt:
                print("Interrupted the conversion. Terminating workers")        
    return "Finished DICOM 2 Nii transformation :)"


parallel_DICOM_2_nii(type=type_im, dicom_df_path=pth, outdir=outdir, n_workers=n_works)
