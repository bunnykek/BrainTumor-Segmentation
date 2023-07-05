# BrainTumor-Segmentation
![title](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/prediction4.png)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bunnykek/BrainTumor-Segmentation/blob/main/BrainTumorSegmentationNotebook.ipynb)

### 0. Dataset
**The dataset has been sourced from The BRATS 2021 (Brain Tumor Segmentation) contest.**
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

1. **Purpose**: The main purpose of the dataset is to promote the development of automated methods for tumor segmentation, which is the process of identifying and delineating tumor regions in brain images.

2. **Image Modalities**: The dataset includes multimodal brain MRI (Magnetic Resonance Imaging) scans, which consist of `four` different imaging modalities: `T1-weighted`, `T1-weighted with gadolinium contrast enhancement (T1Gd)`, `T2-weighted`, and `Fluid Attenuated Inversion Recovery (FLAIR)`. Each modality provides different types of information about the brain structure and tumor characteristics.

3. **Tumor Types**: The dataset covers `four` main types of brain tumors: Glioma, Astrocytoma, Oligodendroglioma, and Glioblastoma.

**How the MRI scans are stored digitally and how to use them in python?**
- A **.nii** file, also known as a NIfTI file, is a common format used in neuroimaging research to store and exchange brain imaging data.
- It stores both the image data and the associated metadata in a single file, making it more convenient to work with.
- To use .nii files in Python, you can use the `nibabel` library.


---

![variants.png](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/variants_slices.png)

- Due to limited computer resources we will be considering only 3 imaging variants for input **(flare, t1ce, t2)**.
- Also t1 image variant has the least contrast around the tumor.


### 1. Pre Processing the data
1.  Now we have 4 image variants(flare, t1ce, t2, seg) for a data set.
2.  The first 3 variants **(flare, t1ce, t2)** will me merged into a single numpy data structure. *(Think it of as a image having 3 channels for each pixel)*
3.  4th one(seg) will be used as the desired output format for the model.*(It represents the segments of a tumor)*
4.  Also the images will be cropped from `240x240x240` to `128x128x128` to save space and time.         
> **(flare + t1ce + t2) = merged.npy** `shape:(128, 128, 128, 3)`

### 2. Training
![unet](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/unet_arch.png)
**UNet architecture**           
1. First sight, it has a “U” shape. The architecture is symmetric and consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by transposed 2d convolutional layers.
2. The U-Net architecture is commonly used for brain tumor segmentation because it has proven to be effective in handling semantic segmentation tasks, where the goal is to classify each pixel in an image into different categories.
- https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
- https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d



---


- The model is trained via Nvidia 1660Ti GPU, with a epoch of 20 and image batch size 1(due to memory limit).
- It took around 6 hours to train.
- The model accuracy has come up to be ~98%.


### 3. Accuracy and loss graphs
![train_val_accurac](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/train_val_accuracy.png)

![train_val_loss](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/train_val_loss.png)

### 5. Some predicted results:
![p1](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/prediction1.png)
![p2](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/prediction2.png)
![p3](https://github.com/bunnykek/BrainTumor-Segmentation/blob/main/assets/prediction3.png)

### References
1. https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
2. https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
3. https://github.com/bnsreenu/python_for_microscopists
4. https://www.ibm.com/topics/convolutional-neural-networks
5. https://www.analyticsvidhya.com/blog/2022/10/image-segmentation-with-u-net/
