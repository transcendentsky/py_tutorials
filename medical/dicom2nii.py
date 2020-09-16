# pip install dicom2nifti

# dicom2nifti [-h] [-G] [-r] [-o RESAMPLE_ORDER] [-p RESAMPLE_PADDING] [-M] [-C] [-R] input_directory output_directory

import dicom2nifti
import os

dicom_dir = "D://media/sample/"

for d in os.listdir(dicom_dir):
    print("DIR: ", d)
    if os.path.isdir(os.path.join(dicom_dir, d)):
        print("Converting...")
        nii_name = os.path.join(dicom_dir, d+".nii.gz")
        dicom2nifti.dicom_series_to_nifti(os.path.join(dicom_dir, d), nii_name, reorient_nifti=True)

# dicom2nifti.convert_directory()