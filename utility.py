import numpy as np

def load_img(img_dir, img_list):
    images=[]
    for image_name in img_list:    
        if (image_name.split('.')[-1] == 'npz'):
            image = np.load(img_dir+image_name)["arr_0"]
            images.append(image)
    images = np.array(images)
    return(images)



def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size