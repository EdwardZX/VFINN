# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import numpy as np
import os
import torch
from models.croco import CroCoNet
from models.croco_downstream import CroCoDownstreamPIV
from models.head_downstream import PixelwiseTaskWithDPT
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose,Resize
from stereoflow.datasets_flow import flowToColor
import matplotlib.pylab as plt
import torch.nn.functional as F
from SPTFlowTrainerDPTwithDiffusionLoss import *

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')


TAG_FLOAT = 202021.25

def read_flo_file(filename):
    with open(filename, 'rb') as f:
        # Read the header
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Magic number incorrect. Got {magic}, expected 202021.25")

        # Read width and height
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # Read the data
        data = np.fromfile(f, np.float32, count=2 * width * height)

        # Reshape data into 2D flow
        flow = np.resize(data, (height, width, 2))
        
    return flow


def get_pred_gt_flow(model,filepath, filename, model_name, img_format = '.tif',
                            model_path = './model_dir', model_pretrained = 'pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth',
                            save_plot_path = './plot_bsa_png',
                            is_spt = None):
    # filename = SQG_00001
    f1 = os.path.join(filepath, filename + '_img1' + img_format)
    f2 = os.path.join(filepath, filename + '_img2' + img_format)
    f_flow_gt = os.path.join(filepath, filename + '_flow.flo')
    
    model_save_name_tag = model_name.split('-')[0]
    
    
    os.makedirs(os.path.join(save_plot_path, filepath.split('/')[-1]), exist_ok=True)
    save_plot_path = os.path.join(save_plot_path, filepath.split('/')[-1])
    save_name_prefix = ('_').join([filename, model_save_name_tag])
    save_name = os.path.join(save_plot_path, save_name_prefix + '_tiled_combined_velocity.png') if is_spt is not None else os.path.join(save_plot_path, save_name_prefix + '_tiled_combined_velocity_filed.png')
    # load 224x224 images and transform them to tensor
    
    
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(device, non_blocking=True)


    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    im1 = trfs(Image.open(f1).convert('RGB'))
    im2 = trfs(Image.open(f2).convert('RGB'))
    C, H, W = im1.shape
    win_height, win_width = 224, 224

    do_change_scale =  H < win_height or W < win_width
    if do_change_scale: 
        upscale_factor = max(win_height/H, win_width/W)
        new_size = (round(H*upscale_factor),round(W*upscale_factor))
        f_resize_img = Resize(new_size)
        
        im1 = f_resize_img(im1)
        im2 = f_resize_img(im2)    
        
        do_change_scale = torch.tensor([new_size[1] / W, new_size[0] / H])

    im1 = im1.to(device, non_blocking=True).unsqueeze(0)
    im2 = im2.to(device, non_blocking=True).unsqueeze(0)
    
    # load model, SQGDiffOnlyv1 SQG_Ld
    
    with torch.inference_mode():
        pred, _ = batch_tiled_pred(model, im1, im2, overlap=0.1)


    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy() # why the two orientation mirrored?
    
    
    # vx, vy = pred[..., 0], pred[..., 1]
    fig, axs = plt.subplots()
    # Plot the predicted flow
    axs.imshow(flowToColor(pred))
    axs.axis('off')
    # axs[0].set_title('Prediction')

    # Add a key for scale
    # axs[1].quiver(vx, vy)
    # axs[1].quiver(vx, -vy) # change correct direction relative to the image representation
    
    # for smoothing to visualized the vector field 



    # step = 20  # Adjust this value to control density (larger step = less dense)
    # axs[1].quiver(vx[::step, ::step], -vy[::step, ::step])

    # axs[1].set_title('Velocity Field')
    # axs[1].set_xlabel('X-axis')
    # axs[1].set_ylabel('Y-axis')
    # axs[1].invert_yaxis()  # Invert y-axis if necessary 
    # axs[1].set_aspect('equal')

    # Save the figure
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    save_name = os.path.join(save_plot_path, filename + '_flow.npy')
    np.save(save_name, pred)
    plt.close()



if __name__ == "__main__":
    ## to change
    ## can give a dict
    filenames = {
    'BSA_0_2_0019_piv': 'BSA_200mM_kcl',
    'BSA_0_2_0021_piv': 'BSA_200mM_kcl',
    'bsa_qd_30_0014_piv': 'BSA_200mM_kcl',
    'bsa_qd_30_res_piv': 'BSA_200mM_kcl',
    'BSA_QD_resonant_0001_piv': 'BSA_200mM_kcl',
    'BSA_QD_resonant_0004_piv': 'BSA_200mM_kcl',
    'BSA_QD_resonant_0005_piv': 'BSA_200mM_kcl',
    'BSA-transferrinEqual_Kcl_200_0003_piv': 'BSA_200mM_kcl_1_1_transferrin',
    'BSA-transferrinEqual_Kcl_200_0006_piv': 'BSA_200mM_kcl_1_1_transferrin',
    'bsa_qd_60_0001_piv': 'BSA_200mM_kcl_60mg_ml',
    'bsa_qd_60_0006_piv': 'BSA_200mM_kcl_60mg_ml',
    'BSA_U_QD_piv': 'BSA_200mM_kcl_U',
    'BSA_dex_0_400kcl_0006_piv': 'BSA_400mM_kcl',
    'BSA_dex_0_400kcl_0010_piv': 'BSA_400mM_kcl',
    'BSA_dex_0_400kcl_0011_piv': 'BSA_400mM_kcl',
    'BSA_dex_0_400kcl_0015_piv': 'BSA_400mM_kcl',
    'BSA_dex_0_400kcl_0016_piv': 'BSA_400mM_kcl',
    'bsa_qd_30_kcl_0006_piv': 'BSA_400mM_kcl',
    'bsa_qd_30_kcl_0007_piv': 'BSA_400mM_kcl',
    'BSA_ac_1_1_400kcl_0003_piv': 'BSA_400mM_kcl_ac_1_1',
    'BSA_ac_1_1_400kcl_0006_piv': 'BSA_400mM_kcl_ac_1_1'
}
    
    # filenames = {'bsa_qd_30_kcl_0006_piv': 'BSA_400mM_kcl',}
    datapath = './assets/piv_bsa_samples/'

    modes = 'Ldp'
    

    for piv_folder, tags in filenames.items():
        files = os.listdir(os.path.join(datapath,piv_folder))
    # Filter for image files
        img_files = [f for f in files if f.endswith(('_img1.tif'))]
        
        # Sort the image files
        img_files.sort()
        file_numbers = [f'{i:05d}' for i in range(1, len(img_files))] 
        model_name = tags + '_' + piv_folder + '_' + 'Ldp-noise-0.0-1.pth'
        model = load('./model_dir', model_name, 'pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth').to(device)
        for num in file_numbers:
            filename = f"{piv_folder}_{num}"
            filepath = os.path.join(datapath,piv_folder)
            get_pred_gt_flow(model,filepath, filename, model_name)

