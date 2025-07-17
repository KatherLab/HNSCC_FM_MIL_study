import numpy as np
import pydicom as pyd
import pandas as pd
from pathlib import Path
#Modified from the MIRP package: https://github.com/oncoray/mirp by Zwnaenburg et al.

def get_pydicom_meta_tag(dcm_seq, tag, tag_type=None, default=None, test_tag=False):
    """
    Processes the information on a given DICOM meta-tag specified by (0xNNNN,0xMMMM) from a DICOM sequence.
    dcm_seq: Pydicom dataset,
    tag: (0xNNNN,0xMMMM) form of the DICOM tag to lookup
    default: value to input in case the tag returns empty
    test: bool, returns false if the tag is not present in the dicom sequence
    """
    # Reads dicom tag
    # Initialise with default
    tag_value = default
    # Parse raw data
    try:
        tag_value = dcm_seq[tag].value
    except KeyError:
        if test_tag:
            return False
        else:
            pass

    if test_tag:
        return True

    if isinstance(tag_value, bytes):
        tag_value = tag_value.decode("ASCII")

    # Find empty entries
    if tag_value is not None:
        if tag_value == "":
            tag_value = default

    # Cast to correct type (meta tags are usually passed as strings)
    if tag_value is not None:

        # String
        if tag_type == "str":
            tag_value = str(tag_value)

        # Float
        elif tag_type == "float":
            tag_value = float(tag_value)

        # Multiple floats
        elif tag_type == "mult_float":
            tag_value = [float(str_num) for str_num in tag_value]

        # Integer
        elif tag_type == "int":
            tag_value = int(tag_value)

        # Multiple floats
        elif tag_type == "mult_int":
            tag_value = [int(str_num) for str_num in tag_value]

        # Boolean
        elif tag_type == "bool":
            tag_value = bool(tag_value)

        elif tag_type == "mult_str":
            tag_value = list(tag_value)

    return tag_value


def get_image_type(image_file):
    """
    Function that determines the image format
    """
    # Determine image type
    if image_file.lower().endswith((".dcm", ".ima")):
        image_file_type = "dicom"
    elif image_file.lower().endswith((".nii", ".nii.gz")):
        image_file_type = "nifti"
    elif image_file.lower().endswith(".nrrd"):
        image_file_type = "nrrd"
    else:
        image_file_type = "dicom"

    return image_file_type

def get_basic_metadata(img_file: Path= None, dirs: int = None):
    """
    Function that returns the basic metadata of the dicoms such as modality, scanner, manufacturer etc...
    Args:
    img_file: Path, absolute path to dicom
    dirs: int, amount of .dcm files in the directory itself
    """
    if img_file is not None:
        # Determine image type
        img_file_type = get_image_type(img_file)

        if img_file_type == "dicom":
            # Load dicom file
            dcm = pyd.dcmread(img_file, stop_before_pixels=True, force=True)
        else:
            dcm = None
    if dcm is not None:
        
# Name:
        # Modality
        modality = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str")
        # Instance number
        instance_number = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0013), tag_type="int", default=-1)
        #Study description
        study_desc = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008,0x1030), tag_type="str", default="")	
        # Scanner type
        scanner_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x1090), tag_type="str", default="")
        # Scanner manufacturer
        manufacturer = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0070), tag_type="str", default="")
        # Slice thickness
        spacing_z = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0050), tag_type="float", default=np.nan)
        # Pixel spacing
        pixel_spacing = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0030), tag_type="mult_float")
        #Time of image acquisition
        time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0033), tag_type="float", default=np.nan)
        
        if pixel_spacing is not None:
            spacing_x = pixel_spacing[0]
            spacing_y = pixel_spacing[1]
        else:
            spacing_x = spacing_y = np.nan
        #study comments:
        width=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028,0x0010))
        height=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028,0x0011))
        dims= [dirs,height,width]
        study_comments = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008,0x103E), tag_type="str", default="None")
        orient = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020,0x0037), tag_type="mult_int")
        meta_data = pd.Series({
                               "modality": modality,
                               "Acquisition_time": time,
                               "study" : study_comments,
                               "Description" : study_desc,
                               "orientation": orient,
                               "series_dim":dims,
                               "instance_number": instance_number,
                               "scanner_type": scanner_type,
                               "manufacturer": manufacturer,
                               "spacing_z": spacing_z,
                               "spacing_y": spacing_y,
                               "spacing_x": spacing_x})
    else:
        meta_data = pd.Series({
                               "modality":"",
                               "Acquisition_time":"",
                               "study" :"",
                               "Description" : "",
                               "orientation": "",
                               "series_dim":"",
                               "instance_number": -1,
                               "scanner_type": "",
                               "manufacturer": "",
                               "spacing_z": np.nan,
                               "spacing_y": np.nan,
                               "spacing_x": np.nan})

    return meta_data

def get_basic_ct_meta_data(image_file: Path =None):
    """
    Function that returns a Pandas Series with CT metadata from a DICOM file
    image_file: Path, absolute path to image files
    return: pandas series
    """
    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pyd.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"image_type": "",
                           "kvp": np.nan,
                           "kernel": "",
                           "agent": "",
                           "Contrast_Enhanced" :"",
                           "Tube current (mA)":np.nan,
                           "Radiation exposure (mA)" : np.nan})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "CT":

            # Image type
            image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="str", default="")

            # Peak kilo voltage output
            kvp = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0060), tag_type="float", default=np.nan)

            # Convolution kernel
            kernel = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1210), tag_type="str", default="")

            # Contrast/bolus agent
            contrast_agent = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0010), tag_type="str", default="")
            current = get_pydicom_meta_tag(dcm_seq=dcm,tag=(0x0018, 0x1151),tag_type="float", default=np.nan)

            rad_exposure = get_pydicom_meta_tag(
            dcm_seq=dcm,
            tag=(0x0018, 0x1152),
            tag_type="float", default=np.nan)
            # Phase of contrast
            #Contrast start:
            #Start of contrast injection
            meta_data = pd.Series({"image_type": image_type,
                                   "kvp": kvp,
                                   "kernel": kernel,
                                   "agent": contrast_agent,
                                   "Contrast_Enhanced" :contrast_agent,
                                   "Tube current (mA)": current,
                                   "Radiation exposure (mA)":rad_exposure})

    meta_data.index = "ct_" + meta_data.index

    return meta_data

def get_basic_mr_meta_data(image_file:Path=None):
    """
    Function that returns a Pandas Series with MRI metadata from a DICOM file
    image_file: Path, absolute path to image files
    return: pandas series
    """
    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pyd.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"image_type": "",
                           "scanning_sequence": "",
                           "scanning_sequence_variant": "",
                           "scanning_sequence_name": "",
                           "scan_options": "",
                           "acquisition_type": "",
                           "repetition_time": np.nan,
                           "echo_time": np.nan,
                           "echo_train_length": np.nan,
                           "inversion_time": np.nan,
                           "trigger_time": np.nan,
                           "magnetic_field_strength (T)": np.nan,
                           "Contrast Enhanced": ""})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "MR":

            # Image type
            image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="str", default="")

            # Scanning sequence
            scanning_sequence = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0020), tag_type="str", default="")

            # Scanning sequence variant
            scanning_sequence_variant = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0021), tag_type="str", default="")

            # Sequence name
            scanning_sequence_name = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0024), tag_type="str", default="")

            # Scan options
            scan_options = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0022), tag_type="str", default="")

            # Acquisition type
            acquisition_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0023), tag_type="str", default="")

            # Repetition time
            repetition_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0080), tag_type="float", default=np.nan)

            # Echo time
            echo_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0081), tag_type="float", default=np.nan)

            # Echo train length
            echo_train_length = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0091), tag_type="float", default=np.nan)

            # Inversion time
            inversion_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0082), tag_type="float", default=np.nan)

            # Trigger time
            trigger_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1060), tag_type="float", default=np.nan)

            # Contrast/bolus agent
            contrast_agent = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0010), tag_type="str", default=np.nan)

            # Magnetic field strength
            magnetic_field_strength = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0087), tag_type="float", default=np.nan)

            meta_data = pd.Series({"image_type": image_type,
                                   "scanning_sequence": scanning_sequence,
                                   "scanning_sequence_variant": scanning_sequence_variant,
                                   "scanning_sequence_name": scanning_sequence_name,
                                   "scan_options": scan_options,
                                   "acquisition_type": acquisition_type,
                                   "repetition_time": repetition_time,
                                   "echo_time": echo_time,
                                   "echo_train_length": echo_train_length,
                                   "inversion_time": inversion_time,
                                   "trigger_time": trigger_time,
                                   "magnetic_field_strength (T)": magnetic_field_strength,
                                   "Contrast_Enhanced": contrast_agent})

    meta_data.index = "mr_" + meta_data.index

    return meta_data
