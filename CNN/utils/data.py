import nibabel as nib
import numpy as np

def load_niigz(pid, ext="seg4_la_4ch", dataDir="/Users/kexiao/Data/ukbb/niigz",
               verbose=False):
    data = nib.load(f"{dataDir}/{pid}/{ext}.nii.gz").get_fdata().squeeze()
    if data.ndim == 3:
        data = data.transpose(2,0,1)
    elif data.ndim == 4:
        data = data.transpose(2,3,0,1)
    if verbose:
        print(f"loaded {pid}/{ext}.nii.gz: ", data.shape)
    return data
