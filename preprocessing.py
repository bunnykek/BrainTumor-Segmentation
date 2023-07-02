import glob
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

if __name__=='__main__':
    if not os.path.exists(os.path.join("processed_data")):
        os.makedirs(os.path.join("processed_data", "input"))
        os.makedirs(os.path.join("processed_data", "output"))

    scaler = MinMaxScaler()
    flair_files = glob.glob("BraTS2021_Training_Data/BraTS2021_*/BraTS2021_*_flair.nii.gz")
    t1_files = glob.glob("BraTS2021_Training_Data/BraTS2021_*/BraTS2021_*_t1.nii.gz")
    t2_files = glob.glob("BraTS2021_Training_Data/BraTS2021_*/BraTS2021_*_t2.nii.gz")
    t1ce_files = glob.glob("BraTS2021_Training_Data/BraTS2021_*/BraTS2021_*_t1ce.nii.gz")
    seg_files = glob.glob("BraTS2021_Training_Data/BraTS2021_*/BraTS2021_*_seg.nii.gz")

    # print(len(flair_files), len(t1_files), len(t2_files), len(t1ce_files), len(seg_files))

    for i in range(len(flair_files)):
        flare_data = nb.load(flair_files[i]).get_fdata()
        t1_data = nb.load(t1_files[i]).get_fdata()
        t2_data = nb.load(t2_files[i]).get_fdata()
        t1ce_data = nb.load(t1ce_files[i]).get_fdata()
        seg_data = nb.load(seg_files[i]).get_fdata()
        
        # Converting datatype from float to int8 because it has only 4 unique values(0, 1, 2, 4)
        seg_data = seg_data.astype(np.int8)
        
        # reassigning 4 -> 3
        seg_data[seg_data==4] = 3 
        # print(np.unique(seg_data))  #(0, 1, 2, 3)

        # cropping out unecessary data
        flare_data = flare_data[40:210,40:210,:]
        t1_data = t1_data[40:210,40:210,:]
        t2_data = t2_data[40:210,40:210,:]
        t1ce_data = t1ce_data[40:210,40:210,:]
        seg_data = seg_data[40:210,40:210,:]
        
        # Normalizing the data to range [0,1]
        flare_data = scaler.fit_transform(flare_data.reshape(-1, flare_data.shape[-1])).reshape(flare_data.shape)
        t1_data = scaler.fit_transform(t1_data.reshape(-1, t1_data.shape[-1])).reshape(t1_data.shape)
        t2_data = scaler.fit_transform(t2_data.reshape(-1, t2_data.shape[-1])).reshape(t2_data.shape)
        t1ce_data = scaler.fit_transform(t1ce_data.reshape(-1, t1ce_data.shape[-1])).reshape(t1ce_data.shape)

        # combining all the 4 image variants into a single image
        merged = np.stack([flare_data, t1_data, t1ce_data, t2_data], axis=-1)
        
        #saving the image if the mask has enough usable data
        _, counts = np.unique(seg_data, return_counts=True)
        if 1 - counts[0]/counts.sum() > 0.01:   #if useful data is atleast 1% in the given volume
            seg_data = to_categorical(seg_data, num_classes=4)
            np.savez_compressed(os.path.join("processed_data", "input", "input_"+str(i).zfill(4)+".npz"), merged)
            np.savez_compressed(os.path.join("processed_data", "output", "output_"+str(i).zfill(4)+".npz"), seg_data)
            print("Done", i)

        
        import splitfolders
        input_folder = 'processed_data'
        output_folder = 'split_data'
        splitfolders.ratio(input_folder, output=output_folder, seed=34353, ratio=(.75, .25), group_prefix=None, move=True)
        os.removedirs('processed_data')


