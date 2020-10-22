# pip install dicom2nifti

# dicom2nifti [-h] [-G] [-r] [-o RESAMPLE_ORDER] [-p RESAMPLE_PADDING] [-M] [-C] [-R] input_directory output_directory

import dicom2nifti
import os

# dicom_dir = "D://media/sample/"
# dicom_dir = "D://media//medical//lsw//benign_65//fpAML_55"
dicom_dir = "D:\media\medical\lsw//benign_65\oncocytoma_10"

# output_dir = "D://media//medical//lsw_trans//benign_65//fpAML_55//"
output_dir = "D:\media\medical\lsw//benign_65\oncocytoma_10"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
error_names = []
for d in os.listdir(dicom_dir):
    print("DIR: ", d)
    if os.path.isdir(os.path.join(dicom_dir, d)):
        nii_name = os.path.join(output_dir, d+".nii.gz")
        if os.path.exists(os.path.join(nii_name)):
            continue
        print("Converting... ", d)
        try:
            dicom2nifti.dicom_series_to_nifti(os.path.join(dicom_dir, d), nii_name, reorient_nifti=False)
        except Exception as e:
            error_names.append(d)
            print(e)

print("Error files: ", error_names)
# dicom2nifti.convert_directory()