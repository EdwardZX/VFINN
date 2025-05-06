import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from SPTFlowTrainerDPTwithDiffusionLoss import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default= 150, #50 75
                    help='Default number of training epochs (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=16, help= '(default: %(default)s)')
parser.add_argument('--lr', type=float, default=5e-4, help= '(default: %(default)s)')
parser.add_argument('--seed', type=int, default=0, help= '(default: %(default)s)')
parser.add_argument('--clip_grad', type=float, default=5.0, help= '(default: %(default)s)')
parser.add_argument('--warmup_steps', type=int, default=150, #400, 2000
                    help='Training warmup steps (default: %(default)s)')
parser.add_argument('--sample_rates', type=float, default=1.0,
                    help='training  samples rates (default: %(default)s)')
parser.add_argument('--dload', type=str, default='./model_dir',
                    help='save_dir (default: %(default)s)')
parser.add_argument('--RESUME', type=bool, default=False,
                    help='The checkpoint is resume train (default: %(default)s)')
parser.add_argument('--is_gt', action='store_true',
                    help='The ground truth of optical flow for evaluation')
parser.add_argument('--is_spt_pt', action='store_true',
                    help='The points set of single particle in a image for tracking')
parser.add_argument('--noise_adding', type=float, default=0.0,
                    help='The external noise is adding noise level (default: %(default)s)')
parser.add_argument('--beta_diffL', type=float, default=1., 
                    help='The weight of diffusion loss component')
parser.add_argument('--beta_photo', type=float, default=1e-5, 
                    help='The filename of photometric loss component')
parser.add_argument('--tilted_overlap', type=float, default=0.05, 
                    help='The tilted_overlap of the crop in the full image')
parser.add_argument('--filename', type=str, default='0473spt_all_pt_Ldp', 
                    help='The filename in data folder')
parser.add_argument('--link_method', type=str, choices=['nn', 'kalman', 'lap'], default='nn',
                    help='Linking method to use for tracking: nearest neighbors (nn), Kalman filter (kalman), or linear assignment problem (lap)')
parser.add_argument('--is_train', action='store_true',
                    help='Flag indicating whether to run in training mode (True) or evaluation mode (False)')


configs = parser.parse_args()
torch.cuda.empty_cache() #cache into zero to avoid other process use the memory space
torch.manual_seed(seed=configs.seed) #once the init seed used, every file would run the same result of the fixed order random numbers

if __name__ == '__main__':
    is_train = configs.is_train
    if configs.filename is not None:
        # searching training number
        saved_name = '-'.join([configs.filename,'noise',str(configs.noise_adding)])
        file_sets = glob.glob(os.path.join(configs.dload, saved_name +"*.pth"))        
        cnt = max(len(file_sets),1) #default network
        if is_train and (not configs.RESUME):
            save_name = '-'.join([saved_name,str(cnt)]) # overlay
        else:
            save_name = '-'.join([saved_name, str(cnt)])
    
    configs.save_name = save_name
    configs.model_name = save_name + '.pth'
    annotation_prefix = ('_').join(configs.filename.split('_')[0:-1]) + '_'
    configs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_datasets, test_datasets = generate_dataset_torch(
                                    MultiPairedImageDataset('./assets/annotations', annotation_prefix + 'annotation.txt',configs=configs),
                                    sample_rates = configs.sample_rates)
    
    if is_train:
        model = train(train_datasets=train_datasets,
                      pretrained_model_path='pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth',
                      configs=configs,
                      test_datasets=test_datasets)
    else:
        model = load(configs.dload, configs.model_name,
                     'pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth').to(configs.device)
        
        ## for getting spt
        if configs.is_spt_pt:
            spt_datasets, _  = generate_dataset_torch(
                                        MultiPairedImageDataset('./assets/annotations',annotation_prefix + 'annotation.txt',configs=configs
                                                                ), train_prob = 1.)
            get_spt_point_pairs(spt_datasets,model,configs, eps=10.0)
            
            if configs.link_method == 'nn':
                get_spt_trackers(configs.dload, configs.save_name +'_all_spots_pairs' + '.csv')
            else:
                get_spt_optim_trackers(configs.dload, configs.save_name +'_all_spots_pairs' + '.csv', 
                                       track_md = configs.link_method, 
                                       link_eps = 10.0)