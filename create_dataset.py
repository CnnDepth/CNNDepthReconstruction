from preprocessing import get_synched_frames, extract_images, process_images
import numpy as np
import h5py
import os


def create_hdf5_dataset(data_dirs, hdf5_filename, data_percent=5.):
    depths = []
    rgbs = []
    
    print("Extracting images...")
    for subdir in data_dirs:
        subsubdirs = list(os.walk(subdir))[0][1]
        for subsubdir in subsubdirs:
            path_to_dir = os.path.join(subdir, subsubdir)
            sync_frames = get_synched_frames(path_to_dir)
            X_part, Y_part = extract_images(path_to_dir, sync_frames, data_percent=data_percent)
            depths += X_part
            rgbs += Y_part
    print("Done.")
    
    print("Processing images...")
    processedDepths, processedRGBs = process_images(depths, rgbs)
    processedDepths = np.array(processedDepths)
    processedRGBs = np.array(processedRGBs)
    print("Done")
    
    print("RGB data shape:", processedRGBs.shape)
    print("Depth data shape:", processedDepths.shape)
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("Depth", data=processedDepths)
        f.create_dataset("RGB", data=processedRGBs)