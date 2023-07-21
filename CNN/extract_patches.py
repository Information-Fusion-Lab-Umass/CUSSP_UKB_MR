
import tqdm
import argparse
import pandas as pd
import numpy as np

from UKBB_utils.ukbb.zip import load_zip_data
from UKBB_utils.utils.filepath import secure_folder

from utils.mr_tracker import get_patch_mapping_hist



def main(args):

    #zip_dir = "/Users/kexiao/Data/ukbb/zips"
    #mask_dir = "/Users/kexiao/Data/ukbb/masks/20208/lax_4ch/bw_ec"

    load_image = lambda x: load_zip_data(f"{args.zip_dir}/{x}_20208_2_0.zip", 
                                         "CINE_segmented_LAX_4Ch", 
                                         return_array=True).transpose(0,2,1).astype(np.float64)
    load_mask = lambda y: np.load(f"{args.mask_dir}/{y}.npy").astype(np.float64)
    
    patch_mapping_kwargs = dict(
                                margins = (32,32,30),
                                image_size = (64, 64),
                                n_frames = 50,
                                boundary_guard = False,
                                load_image = load_image,
                                load_mask = load_mask,
                            )

    csv = pd.read_csv(args.csv)

    secure_folder(args.patch_dir)
    secure_folder(f"{args.patch_dir}/npy64/anchor/og")
    secure_folder(f"{args.patch_dir}/npy64/anchor/la")
    secure_folder(f"{args.patch_dir}/npy64/anchor/pa")


    error_pid_log = open(f"{args.patch_dir}/error_pid.log", "a")
    for pid in tqdm.tqdm(csv.PID, desc="Extracting MR patch"):
        try:
            og_patch, la_patch, pa_patch, _ = get_patch_mapping_hist(pid, **patch_mapping_kwargs)
            np.save(f"{args.patch_dir}/npy64/anchor/og/{pid}.npy", og_patch[0])
            np.save(f"{args.patch_dir}/npy64/anchor/la/{pid}.npy", la_patch[0])
            np.save(f"{args.patch_dir}/npy64/anchor/pa/{pid}.npy", pa_patch[0])
        except:
            error_pid_log.write(f"{pid}\n")

    error_pid_log.close()
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--csv", type=str, help="csv file with PID to extract mitral valve patches.")
    argparser.add_argument("--zip_dir", type=str, help="zip directory for 20208 zip files.")
    argparser.add_argument("--mask_dir", type=str, help="mask directory for segmented masks.")
    argparser.add_argument("--patch_dir", type=str, help="patch directory for saving patches.")

    args = argparser.parse_args()
    main(args)
