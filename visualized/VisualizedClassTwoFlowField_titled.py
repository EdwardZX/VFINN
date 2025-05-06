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


def visualized_pred_gt_flow(filepath, filename, model_name, img_format = '.tif',
                            model_path = './model_dir', model_pretrained = 'pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth',
                            save_plot_path = './plot_flow_png',
                            is_spt = None):
    # filename = SQG_00001
    f1 = os.path.join(filepath, filename + '_img1' + img_format)
    f2 = os.path.join(filepath, filename + '_img2' + img_format)
    f_flow_gt = os.path.join(filepath, filename + '_flow.flo')
    
    model_save_name_tag = model_name.split('-')[0]
    save_name_prefix = ('_').join([filename, model_save_name_tag])
    save_name = os.path.join(save_plot_path, save_name_prefix + '_tiled_combined_velocity.png') if is_spt is not None else os.path.join(save_plot_path, save_name_prefix + '_tiled_combined_gt.png')
    # load 224x224 images and transform them to tensor
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(device, non_blocking=True)


    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    im1 = trfs(Image.open(f1).convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    im2 = trfs(Image.open(f2).convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    
    # load model, SQGDiffOnlyv1 SQG_Ld
    model = load(model_path, model_name, model_pretrained).to(device)
    with torch.inference_mode():
        pred, _ = batch_tiled_pred(model, im1, im2, overlap=0.1)


    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy() # why the two orientation mirrored?
    
    
    if is_spt is not None:
        vx, vy = pred[..., 0], pred[..., 1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the predicted flow
        axs[0].imshow(flowToColor(pred))
        axs[0].axis('off')
        axs[0].set_title('Prediction')

        # Add a key for scale
        # axs[1].quiver(vx, vy)
        axs[1].quiver(vx, -vy) # change correct direction relative to the image representation
        
        axs[1].set_title('Velocity Field')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')
        axs[1].invert_yaxis()  # Invert y-axis if necessary 
        # Plot the ground truth flow
        # # Save the figure
        plt.savefig(save_name,dpi=300, bbox_inches='tight', pad_inches=0)
        return 
    
    flow_gt = read_flo_file(f_flow_gt)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the predicted flow
    axs[0].imshow(flowToColor(pred))
    axs[0].axis('off')
    axs[0].set_title('Prediction')

    # Plot the ground truth flow
    axs[1].imshow(flowToColor(flow_gt))
    axs[1].axis('off')
    axs[1].set_title('Ground Truth')

    # Save the figure
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    ## to change
    ## can give a dict
    suffix_piv = {
        'TestBackstep': '', 
        'TestCylinder': '', 
        'TestDns': '', 
        'TestJhtdb': '',
        'TestSqg':'',
    }

    modes = ['Lp', 'Ld', 'Ldp']
    file_numbers = [f"{i:05d}" for i in range(10)]

    for piv_tags in suffix_piv.keys():
        for mode in modes:
            for num in file_numbers:
                if suffix_piv[piv_tags] == '':
                    filename = f"{piv_tags}_{num}"
                else:
                    filename = f"{piv_tags}_{suffix_piv[piv_tags]}_{num}"
                # filename = ('_').join([piv_tags, suffix_piv[piv_tags],num ])
                # filename = piv_tags + '_' + suffix_piv[piv_tags] + '_' + num
                model_name = 'ClassTwo' + '_' + mode + '-noise-0.0-1.pth'
                filepath = './assets/piv_problem_two_samples/' + piv_tags + '_Dataset_10Imgs'
                visualized_pred_gt_flow(filepath, filename, model_name,
                save_plot_path = './plot_flow_png_classTwo')

