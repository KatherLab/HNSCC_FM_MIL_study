import pandas as pd
import os
from Metadata_parsers import get_image_type,  get_basic_ct_meta_data, get_basic_metadata, get_basic_mr_meta_data

def image_metadata_create(lst_file_pths:list):
    '''
    Function to extract metadata from nested radiology structures.
    Currently supports MRI and CT.
    Inputs: 
    lstfile_pths: lst of strings; list of absolute paths to the first DICOM file of a series
    Outputs: two dataframes, the first for CT metadata and the second for MRI metadata.
    '''
    lst_meta_CT = []
    lst_meta_MRI = []
    for i in lst_file_pths:
        if len(os.listdir(i)) !=0:
            dirs= os.listdir(i)[0] #Metadata should be repeated across .dcm so we just take the first one
            dirs_2 =len(os.listdir(i))  
            dirs = os.path.join(i,dirs)
            if get_image_type(image_file=dirs) == "dicom":
                basic = get_basic_metadata(img_file=dirs, dcm=True, dirs=dirs_2)
                basic["path"] = str(dirs)
                
                if basic.modality == "CT":
                    mod_specs = get_basic_ct_meta_data(image_file=dirs, dcm=True)
                    basic = pd.concat([basic,mod_specs])
                    lst_meta_CT.append(basic)
                if basic.modality == "MR":
                    mod_specs = get_basic_mr_meta_data(image_file=dirs, dcm =True)
                    basic = pd.concat([basic,mod_specs])
                    lst_meta_MRI.append(basic)
            else:
                print("Unsupported format for " + dirs)
    lst_meta_CT= pd.DataFrame(lst_meta_CT) 
    lst_meta_MRI= pd.DataFrame(lst_meta_MRI) 

    if lst_meta_CT.shape[0] == 0:
        return lst_meta_MRI
    if lst_meta_MRI.shape[0] == 0:
        return lst_meta_CT
    else:
        return lst_meta_CT, lst_meta_MRI
