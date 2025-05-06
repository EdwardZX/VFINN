# cpu
import os, csv
import logging
import time, datetime
import numpy as np
import pandas as pd
import math
from utils.tracker import Tracker as optimTracker


# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset,ConcatDataset

# dataloader property
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomCrop

# network
from models.croco import CroCoNet
from models.croco_downstream import CroCoDownstreamPIV, PIVDiffL
from models.head_downstream import PixelwiseTaskWithDPT

from tqdm import tqdm
from collections import defaultdict

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

#write log notes
class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class MyLogger(object, metaclass=SingletonType):
    _logger = None

    def __init__(self,filename,verbosity=1):
        self._logger = logging.getLogger("crumbs")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

        now = datetime.datetime.now()

        fileHandler = logging.FileHandler(filename,"a")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        print("Generate new instance")

    def get_logger(self):
        return self._logger
    
class PairedImageDataset(Dataset):
    def __init__(self, image_folder, configs):
        #image pair
        self.image_folder = image_folder

        #transformers
        pretrained_HW = 224

        self.pretrained_HW = pretrained_HW
        self.random_crop = RandomCrop([self.pretrained_HW,self.pretrained_HW])

        self.is_tiled_eval = False
        self.is_gt = configs.is_gt
        self.is_spt_pt = configs.is_spt_pt
        

        # is_gt and is_spt_pt cannot presever simulatenously
        assert not (self.is_gt and self.is_spt_pt), "is_gt_piv and is_spt_pt cannot be True simultaneously"

        self.noise_std = configs.noise_adding
        self.do_change_scale = None
        # self.trfs = Compose([Resize((pretrained_HW,pretrained_HW)),
        #                      ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
        self.trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
        
        
        # setting 
        if self.is_gt and not self.is_spt_pt:
            is_mode = 'piv'
        elif not self.is_gt and self.is_spt_pt:
            is_mode = 'spt'
        else:
            is_mode = None
        
        self.image_pairs = self._group_images(is_mode)
            
    
    def _group_images(self, is_mode = 'piv'): #
        valid_extensions = ('.tif', '.png', '.jpg', '.jpeg')
        def replace_suffix(s, replaceda_suffix, delimeter = '.'):
            s = s.split(delimeter)
            s[-1] = replaceda_suffix
            return (delimeter).join(s)
        # Group images by their prefix
        images = [img for img in os.listdir(self.image_folder) if img.endswith(valid_extensions)]
        image_groups = {}
        for image in images:
            prefix = ('_').join(image.split('_')[0:-1])
            if prefix not in image_groups:
                image_groups[prefix] = []
            image_groups[prefix].append(image)

        # Convert the dictionary to a list of tuples
        image_pairs = []
        for prefix in sorted(image_groups.keys()):  # Sort the groups by prefix
            group = image_groups[prefix]
            if len(group) == 2:  # Ensure there are pairs and order
                # sort the group element
                group.sort()
                if is_mode == 'piv':
                    piv_gt = prefix + '_flow.flo'
                    assert os.path.exists(os.path.join(self.image_folder, piv_gt)), f"Ground truth file does not exist: {piv_gt}"
                    image_pairs.append((group[0], group[1],piv_gt))
                elif is_mode == 'spt':
                    spt_pt_first = replace_suffix(group[0],'csv')
                    spt_pt_second = replace_suffix(group[1],'csv')
                    assert os.path.exists(os.path.join(self.image_folder, spt_pt_first)) and os.path.exists(os.path.join(self.image_folder, spt_pt_second)), f"point coordinates file does not exist: {spt_pt_first} or {spt_pt_second}"
                    image_pairs.append((group[0], group[1],spt_pt_first, spt_pt_second))

                else:  
                    image_pairs.append((group[0], group[1]))

        
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_names = self.image_pairs[idx]
        
        img1_path = os.path.join(self.image_folder, img_names[0])
        img2_path = os.path.join(self.image_folder, img_names[1]) # image pair loading

        # trfs is transform
        img1 = self.trfs(Image.open(img1_path).convert('RGB'))
        img2 = self.trfs(Image.open(img2_path).convert('RGB'))

        # noisy image
        img1 = self._add_gaussian_noise(img1)
        img2 = self._add_gaussian_noise(img2)
        
        # judge whether the groundtruth
        piv_gt = -1 # 
        if self.is_gt and self.is_tiled_eval:
            piv_gt = self._read_flo_file(os.path.join(self.image_folder, img_names[-1]))
            piv_gt = torch.from_numpy(piv_gt).float().permute(2,0,1) # C, H, W
        if self.is_spt_pt and self.is_tiled_eval:
            spt_first = np.genfromtxt(os.path.join(self.image_folder, img_names[-2]), 
                                            delimiter=',', skip_header=1, dtype=float)
            spt_second = np.genfromtxt(os.path.join(self.image_folder, img_names[-1]), 
                                            delimiter=',', skip_header=1, dtype=float)
            ## if the spt is empty
            if spt_first.size == 0:
                spt_first = np.array([[-1.0, -1.0, -1.0]])
       
            if spt_second.size == 0:
                spt_second = np.array([[-1.0, -1.0, -1.0]])
            
            spt_first = torch.from_numpy(spt_first).float()
            spt_second = torch.from_numpy(spt_second).float()
            
        
        #check the imgs and optical size make smaller image to fixed size, if less than up to pretrained_HW
        C, H, W = img1.shape
        win_height, win_width = self.pretrained_HW, self.pretrained_HW

        do_change_scale =  H < win_height or W < win_width
        if do_change_scale: 
            upscale_factor = max(win_height/H, win_width/W)
            new_size = (round(H*upscale_factor),round(W*upscale_factor))
            f_resize_img = Resize(new_size)
            
            img1 = f_resize_img(img1)
            img2 = f_resize_img(img2)
            if self.is_gt and self.is_tiled_eval:
                piv_gt = f_resize_img(piv_gt)
            if self.is_spt_pt and self.is_tiled_eval:
                # spt_first = self._resize_coordinates(spt_first, (H, W), new_size)
                # spt_second = self._resize_coordinates(spt_second, (H, W), new_size)
                if not torch.all(spt_first == -1):
                    spt_first = self._resize_coordinates(spt_first, (H, W), new_size)
                if not torch.all(spt_second == -1):
                    spt_second = self._resize_coordinates(spt_second, (H, W), new_size)
                
            
            self.do_change_scale = torch.tensor([new_size[1] / W, new_size[0] / H])

        # < crop, scale to
        #  
        
        if self.is_tiled_eval:
            if self.is_gt: # making a tuple
                return img1, img2, piv_gt

            if self.is_spt_pt:
                return img1, img2, spt_first, spt_second
        
        else: # training
            # random crop
            img12 = torch.concatenate([img1, img2],dim=0)
            img12 = self.random_crop(img12)

            return img12[:C], img12[C:]

        return img1, img2 # default 

    
    def _read_flo_file(self, filename):
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
    
    def _resize_coordinates(self,coordinates, original_size, new_size):
        """
        Resize coordinates based on the scaling factor of the image resize.
        
        :param coordinates: List of (x, y) tuples or tensor of shape (N, 2)
        :param original_size: Tuple of (original_height, original_width)
        :param new_size: Tuple of (new_height, new_width)
        :return: Resized coordinates
        """
        orig_h, orig_w = original_size
        new_h, new_w = new_size
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        scale = torch.tensor([1, scale_x, scale_y]) # time x, y
        return coordinates * scale
    
    def _add_gaussian_noise(self, image):
        return image + torch.randn_like(image) * self.noise_std
    
    def set_eval_mode(self):
        self.is_tiled_eval = True
    
    def set_train_mode(self):
        self.is_tiled_eval = False
    
    

#get the full dataloader of multiple folder
def MultiPairedImageDataset(filepath,filename,configs):
    dataloaders = []
    df = pd.read_csv(os.path.join(filepath,filename),delimiter=',',header=None)
    for i in range(df.shape[0]):
        dataloaders.append(PairedImageDataset(df.iloc[i,0], configs))
    return ConcatDataset(dataloaders)

def generate_dataset_torch(full_dataset,train_prob = 0.75, sample_rates = 1.):
    
    if 0 < sample_rates < 1:
        train_sz = int(sample_rates * train_prob * len(full_dataset))
        test_sz = int(sample_rates * (1-train_prob) * len(full_dataset))
        other_sz = len(full_dataset) - train_sz - test_sz
        train_dataset, test_dataset, _ = torch.utils.data.random_split(full_dataset, [train_sz, test_sz, other_sz])
    else:
        train_sz = int(train_prob * len(full_dataset))
        test_sz = len(full_dataset) - train_sz
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_sz, test_sz])
    return train_dataset, test_dataset

def save(model,dload,file_name):
    PATH = dload + '/' + file_name
    if os.path.exists(dload):
        pass
    else:
        os.mkdir(dload)
    torch.save(model.state_dict(), PATH)


def save_checkpoint(ck,dload,file_name):
    ckp_PATH = dload + '/' + file_name
    PATH = ckp_PATH + '/checkpoint_ckpt.pth'
    if os.path.exists(ckp_PATH):
        pass
    else:
        os.mkdir(ckp_PATH)
    torch.save(ck, PATH)

# def save_checkpoint(ck, dload, file_name):
#     ckp_PATH = os.path.join(dload, file_name)
#     PATH = os.path.join(ckp_PATH, 'checkpoint_ckpt.pth')
#     os.makedirs(ckp_PATH, exist_ok=True)
#     filtered_model_state = {
#         k: v for k, v in ck['net'].items() if v.requires_grad
#     }
#     filtered_checkpoint = {
#         "net": filtered_model_state,
#         "optimizer": ck['optimizer'],
#         "epoch": ck['epoch'],
#     }
#     torch.save(filtered_checkpoint, PATH)

# # not working by partial weight

# def save(model, dload, file_name):
#     # only save require grad_True parameters
#     model.train() 
#     PATH = os.path.join(dload, file_name)
#     os.makedirs(dload, exist_ok=True)
#     filtered_state_dict = {
#         k: v for k, v in model.state_dict().items() if v.requires_grad
#     }
#     torch.save(filtered_state_dict, PATH)

def load(dload,file_name, pretrained_model_path):
    num_channels = {'stereo': 1, 'flow': 2}['flow']
    ckpt = torch.load(pretrained_model_path, 'cpu')
    # saved_init parameters

    head = PixelwiseTaskWithDPT() # training DPT module
    head.num_channels = num_channels
    model = CroCoDownstreamPIV(head, **ckpt.get('croco_kwargs', {}), mask_ratio=0.0)
    model.load_state_dict(ckpt['model'], strict=False)

    PATH = dload + '/' + file_name
    model.load_state_dict(torch.load(PATH),strict=True)
    return model

def opticalFlowLoss(tensorFlowForward,
                                tensorFirst,
                                tensorSecond):
    lambda_s = 3.0
    # lambda_c = 0.2
    loss = (warpLoss(tensorFirst, tensorSecond, tensorFlowForward) 
            + lambda_s * secondSmoothnessLoss(tensorFlowForward))
    return loss

def warpForward(tensorInput, tensorFlow):
    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
            1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                tensorFlow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                tensorFlow.size(3))

    Forward_tensorGrid = torch.cat(
            [tensorHorizontal, tensorVertical], 1).to(tensorFlow.device)



    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)
    # to [-1, 1], note that when first point such as [1] -> [255], the v calculated 254
    return torch.nn.functional.grid_sample(
        input=tensorInput,
        grid= (Forward_tensorGrid + tensorFlow).permute(0, 2, 3, 1), # B, [vx,vy], H, W -> B , H, W, [p + vx, p + vy]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)

def charbonnierLoss(x, alpha=0.45, beta=1.0, epsilon=1e-3):
    """Compute the generalized charbonnier loss for x
    Args:
        x(tesnor): [batch, channels, height, width]
    Returns:
        loss
    """
    batch, channels, height, width = x.shape
    normalization = torch.tensor(batch * height * width * channels,
                                 requires_grad=False)

    error = torch.pow(
        (x * torch.tensor(beta)).pow(2) + torch.tensor(epsilon).pow(2), alpha)

    return torch.sum(error) / normalization


# photometric difference
def warpLoss(tensorFirst, tensorSecond, tensorFlow):
    """Differentiable Charbonnier penalty function"""
    tensorDifference = tensorSecond - warpForward(tensorInput=tensorFirst,
                                              tensorFlow=tensorFlow)
    return charbonnierLoss(tensorDifference, beta=255.0)


# 2nd order smoothness loss
def _secondOrderDeltas(tensorFlow):
    """2nd order smoothness, compute smoothness loss components"""
    device = tensorFlow.device
    out_channels = 4
    in_channels = 1
    kh, kw = 3, 3

    filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
    filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
    filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
    filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = torch.FloatTensor(filter_x)
    weight[1, 0, :, :] = torch.FloatTensor(filter_y)
    weight[2, 0, :, :] = torch.FloatTensor(filter_diag1)
    weight[3, 0, :, :] = torch.FloatTensor(filter_diag2)

    uFlow, vFlow = torch.split(tensorFlow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def secondSmoothnessLoss(tensorFlow):
    """Compute 2nd order smoothness loss"""
    delta_u, delta_v = _secondOrderDeltas(tensorFlow)
    return charbonnierLoss(delta_u) + charbonnierLoss(delta_v)


def train(train_datasets, pretrained_model_path, configs, test_datasets = None, shuffle = True):
    model_name = configs.model_name
    logger_name = configs.dload + '/'+ configs.save_name + '.log'
    #load model and fixed the weight of pretrained_model
      # load model
    
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1)
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1)

    # Move them to the same device
    imagenet_mean_tensor = imagenet_mean_tensor.to(configs.device)
    imagenet_std_tensor = imagenet_std_tensor.to(configs.device)

    num_channels = {'stereo': 1, 'flow': 2}['flow']
    ckpt = torch.load(pretrained_model_path, 'cpu')

    head = PixelwiseTaskWithDPT() # training DPT module
    head.num_channels = num_channels
    
    print(ckpt['croco_kwargs'])
    model = CroCoDownstreamPIV(head, **ckpt.get('croco_kwargs', {}), mask_ratio=0.0).to(configs.device)
    model.load_state_dict(ckpt['model'], strict=False)

    ckpt_keys = set(ckpt['model'].keys())

    ## diffusion loss
    diff_discriminator = PIVDiffL(img_channel = 3, patch_size = head.dpt.P_H,
                    enc_embed_dim = ckpt['croco_kwargs']['enc_embed_dim'],
                    img_w = 224, 
    ).to(configs.device)


    # Freeze only the parameters that were in the checkpoint
    for name, param in model.named_parameters():
        if name in ckpt_keys:
            param.requires_grad = False
        else:
            param.requires_grad = True


    optimizer = optim.AdamW(list(model.parameters()) + list(diff_discriminator.parameters()),
                            lr=configs.lr, weight_decay=1e-2)
    # optimizer = optim.AdamW(model.parameters(),
    #                         lr=config.lr, weight_decay=1e-2)
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=configs.batch_size,
                              shuffle=shuffle)
    

    max_iters = configs.epochs * math.ceil(len(train_datasets) / configs.batch_size)
    lr_scheduler = CosineWarmupScheduler(optimizer, warmup=configs.warmup_steps, max_iters =max_iters)
    loss_fn = opticalFlowLoss

    ## init and load checkpoint
    logger = MyLogger.__call__(logger_name).get_logger()
    logger.info('starting training with config:')
    logger.info(configs)
    epochs = configs.epochs
    start_epoch = -1
    
    path_checkpoint = configs.dload + '/' + configs.save_name + '/checkpoint_ckpt.pth'

    #whether load ckpt
    if configs.RESUME and os.path.exists(path_checkpoint):
        # path_checkpoint = config.dload + '/' + config.save_name + '/checkpoint_ckpt.pth'
        checkpoint = torch.load(path_checkpoint, 'cpu')
        model.load_state_dict(checkpoint['net'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.last_epoch = (start_epoch+1) * len(train_loader)

    for i in range(start_epoch + 1, epochs):
        model.train()
        diff_discriminator.train()
        epoch_loss = 0
        diff_loss = 0
        photo_loss = 0
        for t, X in enumerate(train_loader):
            im1 = X[0].to(configs.device)
            im2 = X[1].to(configs.device)
             
            pred, out1, out2 = model(im1, im2)
            decoded_image1, decoded_image2 = model.decode_to_image(out1, im1,
                                                              out2, im2,
                                                              imagenet_mean_tensor,
                                                              imagenet_std_tensor)                                                    
            loss_photo = (1e-3*loss_fn(tensorFlowForward = pred, tensorFirst = decoded_image2, tensorSecond = decoded_image1) + loss_fn(tensorFlowForward = pred, tensorFirst = im2, tensorSecond = im1))
            
            # diffusion loss
            if configs.beta_diffL > 0:
                diff_out = model.encode_image_pairs(im1, im2)
                patch_target, patch_target_pred = model.patchify_warpForward(tensorTarget = im1,
                                        tensorInput = im2,
                                        tensorFlow = pred)
                
                loss_diff = diff_discriminator(patch_target, patch_target_pred, diff_out[0], 
                    )
                diff_loss += loss_diff.item()

                loss = configs.beta_photo * loss_photo + configs.beta_diffL * loss_diff
            else:
                loss = loss_photo
            # forward training
            # loss = loss_fn(tensorFlowForward = pred, tensorFirst = decoded_image1, tensorSecond = decoded_image2)
            # loss = (1e-2*loss_fn(tensorFlowForward = pred, tensorFirst = decoded_image2, tensorSecond = decoded_image1) + loss_fn(tensorFlowForward = pred, tensorFirst = im2, tensorSecond = im1)
            # + config.beta_diffL * loss_diff)
            # loss = loss_fn(tensorFlowForward = pred, tensorFirst = im2, tensorSecond = im1) # i dont know why change direction, but work
            epoch_loss += loss.item()
            photo_loss += loss_photo.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), configs.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            #print the loss function
            if (t+1) % 1000 == 0:
                print('epoch {}: {:d}/{:d}: loss {:4f}, photometric loss {:4f}, diffusion loss {:4f} ,and learning rate {:4f}'
                      .format(i,t, len(train_loader),epoch_loss / (t+1), 
                              photo_loss / (t + 1),
                              diff_loss / (t + 1),
                              lr_scheduler.get_lr(epoch=lr_scheduler.last_epoch)))

        print('epoch {}: loss {:4f}, photometric loss {:4f}, diffusion loss {:4f}, and learning rate {:4f}'
              .format(i, epoch_loss / len(train_loader),
                       photo_loss / len(train_loader),
                      diff_loss / len(train_loader),
                      configs.lr * lr_scheduler.get_lr_factor(epoch=lr_scheduler.last_epoch)))
        logger.info('epoch {}: loss {:4f}, photometric loss {:4f}, diffusion loss {:4f}, and learning rate {:4f}'.format(i, epoch_loss / len(train_loader), photo_loss / len(train_loader), diff_loss / len(train_loader),
        configs.lr * lr_scheduler.get_lr_factor(epoch=lr_scheduler.last_epoch)))

        if (i+1) % 5 == 0: #save every 5 epoch 
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": i,
            }
            save_checkpoint(checkpoint, configs.dload, configs.save_name)
            if test_datasets is not None:
                logger.info(evaluate(test_datasets,model,configs))
            print('checkpoint {} saved'.format(i))
    
    #save final model
    save(model, configs.dload, model_name)
    return model



@torch.no_grad()
def batch_tiled_pred(model, img1, img2, gt = None, crop = (224, 224),
               overlap=0.5, conf_mode='conf_expsigmoid_10_5',):
                     
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged

    C = 2 # for optical flow
    B, _, H, W = img1.shape 
    device = img1.device

    win_height, win_width = crop[0], crop[1]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1)
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1)

    # Move them to the same device
    imagenet_mean_tensor = imagenet_mean_tensor.to(device)
    imagenet_std_tensor = imagenet_std_tensor.to(device)

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True
    
    def create_sigmoid_edge_mask(height, width, edge_width=7, steepness=0.5, mask_scale = 10):
        y, x = np.mgrid[:height, :width]
        # Create distance maps from each edge
        left = x
        right = width - x - 1
        top = y
        bottom = height - y - 1
        
        # Combine distances
        distance = np.minimum.reduce([left, right, top, bottom])
        
        # Apply sigmoid function
        mask = 0.5 - 1 / (1 + np.exp(-steepness * (distance - edge_width)))
        mask[mask<=0] = 0.
        mask = mask * mask_scale
        
        return torch.from_numpy(mask).float()


    # keep track of weighted sum of prediction*weights and weights
    beta, betasigmoid = map(float, conf_mode[len('conf_expsigmoid_'):].split('_'))
    predconf = create_sigmoid_edge_mask(win_height, win_width, edge_width=7).to(device)
    predconf = predconf.expand(B, win_height, win_width)

    accu_pred = img1.new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
    accu_conf = img1.new_zeros((B, H, W)) + 1e-16 # accumulate the weights 

    criterion = nn.L1Loss() if gt is not None else opticalFlowLoss
    tiled_losses = []
    
    #define mask
    
    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        im1_crop, im2_crop = _crop(img1,sy1,sx1), _crop(img2,sy2,sx2)
        pred, out1, out2 =  model(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))
        decoded_image1, decoded_image2 = model.decode_to_image(out1, im1_crop,
                                                              out2, im2_crop,
                                                              imagenet_mean_tensor,
                                                              imagenet_std_tensor)        
        if gt is not None: 
            gt_crop = _crop(gt,sy1,sx1)
            tiled_losses.append(criterion(pred, gt_crop).item())
        else:
            tiled_losses.append(criterion(pred, decoded_image1, decoded_image2).item())
                
        conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5))    
        accu_pred[...,sy1,sx1] += pred * conf[:, None,:,:]
        accu_conf[...,sy1,sx1] += conf
        
    pred = accu_pred / accu_conf[:, None,:,:]
    
    return pred, torch.mean(torch.tensor(tiled_losses))

def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)

def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    return img[:,:,sy,sx]


def evaluate(test_datasets,model,configs):
    # set evaluation mode
    for dataset in test_datasets.dataset.datasets:
        dataset.set_eval_mode()



    if dataset.is_spt_pt:
        test_loader = DataLoader(dataset=test_datasets,
                              batch_size=configs.batch_size,
                              shuffle=True,
                              collate_fn = custom_collate_fn)
    else:
        test_loader = DataLoader(dataset=test_datasets,
                              batch_size=configs.batch_size,
                              shuffle=True)
    
    for i in range(1):
        model.eval()
        epoch_loss = 0
        for t, X in enumerate(test_loader):

            im1 = X[0].to(configs.device)
            im2 = X[1].to(configs.device)
            if configs.is_gt:
                _, loss = batch_tiled_pred(model, im1, im2, X[2].to(configs.device))
            else:
                _, loss = batch_tiled_pred(model, im1, im2)

            epoch_loss += loss

            if (t + 1) % 1000 == 0:
                print('evaluation {}: {:d}/{:d}: {} loss {:4f}'
                      .format(i, t, len(test_loader), 
                              'AEE' if configs.is_gt else 'photometric', epoch_loss / (t + 1)))

        eval_acc = ('evaluation {}: {} loss {:4f}'
                    .format(i, 'AEE' if configs.is_gt else 'photometric', 
                            epoch_loss / len(test_loader),))
        
        # in training model recover
        for dataset in test_datasets.dataset.datasets:
            dataset.set_train_mode()
        return eval_acc

def match_coordinates(set1, set2, eps=4.5, padding = -1.):
    # Calculate pairwise distances
    # B x N x D, B x M x D
    # 'donot_use_mm_for_euclid_dist' can get right answer, but slower, if default, eps shoul higher?
    # so can we set distance equals eps * 4, experiment 1.03 ~ 2.5 than 4.83
    # distances = torch.cdist(set1, set2,compute_mode='donot_use_mm_for_euclid_dist')

    B, N, _ = set1.shape

    # low precision distance
    distances = torch.cdist(set1, set2)
    matched_indices = torch.argmin(distances, dim=-1)

    # Apply epsilon threshold, not using this distance
    query_batch = torch.arange(B).unsqueeze(1).expand(B, N)  # Shape: (B, N)
    set2_find = set2[query_batch, matched_indices, :]
    
    ## # Compute high presion the Euclidean distance
    min_distances = torch.sqrt(torch.sum((set1 - set2_find)**2, dim=-1))

    # Compute the Euclidean distance
    valid_matches = min_distances <= eps

    # Set invalid matches to -1
    set2_find[~valid_matches] = padding
    return set2_find

def gather_points(A, q, padding = -1.):
    """
    Gathers points from tensor A based on coordinates provided in q.

    Parameters:
    - A (torch.Tensor): Input tensor of shape (B, 2, H, W)
    - q (torch.Tensor): Query tensor of shape (B, N, 2) containing (x, y) coordinates

    Returns:
    - torch.Tensor: Output tensor of shape (B, 2, N) containing the gathered points
    """
    B, C, _, _ = A.shape
    B_q, N, _ = q.shape
    
    assert B == B_q, "Batch size of A and q must match."
    assert C == 2, "The second dimension of A must be 2."

    # Extract x and y coordinates and convert to long for indexing
    x = q[:, :, 0]  # Shape: (B, N)
    y = q[:, :, 1]  # Shape: (B, N)
    
    # # if the position value is negative to (0,0)
    # x = torch.clamp(x, min=0)
    # y = torch.clamp(y, min=0)
    padding_mask_x = x == padding
    padding_mask_y = y == padding

    # Clamp non-padding negative values to 0
    x = torch.where(padding_mask_x, x, torch.clamp(x, min=0))
    y = torch.where(padding_mask_y, y, torch.clamp(y, min=0))
    

    # Clamp the coordinates to ensure they are within bounds
    x, y = torch.floor(q[:, :, 0]).long(), torch.floor(q[:, :, 1]).long()


    # Create a batch index tensor
    # Shape: (B, 1) -> (B, N) after expansion
    batch_indices = torch.arange(B).view(B, 1).expand(-1, N).to(A.device)  # Ensure it's on the same device

    # Use advanced indexing to gather the points
    # A has shape (B, 2, H, W)
    # B x N querys
    # We index A[batch_indices, :, y, x] to get shape (B, N, 2)
    gathered = A[batch_indices, :, y, x]  # Shape: (B, N, 2)

    return gathered


def batch_paired_spt(pred, pt1, pt2, eps = 4.5, padding = -1):
    B, _, H, W = pred.shape
    device = pred.device
    def CooridnateFlowForward():
        tensorHorizontal = torch.arange(W).view(
                1, 1, 1, W).expand(B, -1, H, -1).float()
        tensorVertical = torch.arange(H).view(
                1, 1, H, 1).expand(B, -1, -1, W).float()

        return torch.cat(
                [tensorHorizontal, tensorVertical], 1).to(device)

    #warp
    spt_cooridnate = CooridnateFlowForward() # B x D x H x W, D = 2 for xy
    I1_map_coordinate_I2 = warpForward(tensorInput = spt_cooridnate, tensorFlow = pred)
    
    # query the point (B,N,2) in (B, 2, H, W)
    pred_pt2 = gather_points(I1_map_coordinate_I2, pt1, padding = padding)
    pt_paired = match_coordinates(pred_pt2, pt2, eps=eps, padding = padding)
    
    # return torch.cat([pt1, pt_paired, pred_pt2], dim = -1)
    return pt1, pt_paired, pred_pt2

def pad_sequence(point_sets, batch_first=False, padding_value=-1):
    # Get maximum length
    d_dim = point_sets[0].shape[-1]
    max_size = max(len(ps) for ps in point_sets)
    # Determine dimensions for the output tensor
    out_dims = (len(point_sets), max_size, d_dim) if batch_first else (max_size, len(point_sets), d_dim)
    # Initialize the output tensor with the padding value
    out_tensor = torch.full(out_dims, padding_value).float()
    # Pad point_sets
    for i, ps in enumerate(point_sets):
        length = ps.shape[0]
        if batch_first:
            out_tensor[i, :length, :] = ps
        else:
            out_tensor[:length, i, :] = ps
    return out_tensor

def custom_collate_fn(batch):
    # Assume each sample is a point set of (x_i, y_i) points
    images1, images2, point_sets1, point_sets2 = zip(*batch)
    return (torch.stack(images1, dim=0), torch.stack(images2, dim=0), 
            pad_sequence(point_sets1, batch_first=True), pad_sequence(point_sets2, batch_first=True))

def write_batches_to_csv(pt1_sort, pt2_sort, pt2_pred_sort, filename='output.csv', mode='a'):
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if it's the first write operation
        if mode == 'w':
            header = ['t1', 'x1', 'y1', 't2', 'x2', 'y2', 't2_pred', 'x2_pred', 'y2_pred']
            writer.writerow(header)
            # return
        
        B, N, C = pt1_sort.shape
        assert C == 3, f"Expected 3 channels of t, x, y, but got {C}"
        
        # Reshape and concatenate all data at once
        all_data = torch.cat([
            pt1_sort.reshape(B*N, C),
            pt2_sort.reshape(B*N, C),
            pt2_pred_sort.reshape(B*N, C)
        ], dim=1)
        
        # Convert to numpy
        all_data_np = all_data.cpu().numpy()

        mask1 = (all_data_np[:, 1] == -1) & (all_data_np[:, 2] == -1)
        mask2 = (all_data_np[:, 0] == -1)
        final_mask = ~(mask1 | mask2)
        # Apply the mask to filter the data
        filtered_data = all_data_np[final_mask]
        writer.writerows(filtered_data)


@torch.no_grad()
def get_spt_point_pairs(spt_datasets,model,configs, eps = 4.5, padding = -1.):
    # set evaluation mode
    for dataset in spt_datasets.dataset.datasets:
        dataset.set_eval_mode()
    
    assert dataset.is_spt_pt, "test_loader is incorrectly set to spt mode."

    spt_loader = DataLoader(dataset=spt_datasets,
                              batch_size=configs.batch_size,
                              shuffle=False,
                              collate_fn = custom_collate_fn)
    
    model.eval()
    ## writing csv files
    output_file = configs.dload + '/'+ configs.save_name +'_all_spots_pairs' + '.csv'
    # Create the file and write the header
    write_batches_to_csv(torch.empty(0,0,3), torch.empty(0,0,3), torch.empty(0,0,3), 
                         filename = output_file, mode = 'w')

    # for X Y template

    for t, X in enumerate(tqdm(spt_loader, desc="Processing SPT Batches")):
        # img1, img2, pt1, pt2
        im1 = X[0].to(configs.device)
        im2 = X[1].to(configs.device)
        pt1 = X[2].to(configs.device)
        pt2 = X[3].to(configs.device)

        # printed nodes
        # warpForward
        pred, _ = batch_tiled_pred(model, im1, im2)
        pt1_sort, pt2_sort, pt2_pred_sort = batch_paired_spt(pred,pt1[:, :, 1:], pt2[:, :, 1:], 
                                                             eps=eps, padding = padding) #(t, x, y) -> (x, y)
        

        # ds = torch.norm(pt2_pred_sort - pt2_sort, dim=-1)  # Calculate Euclidean distance
        # pt2_sort[ds > JD] = padding
        

        # contained multiple (-1, -1) in first pt
        # recover time
        # if do scale 
        if dataset.do_change_scale is not None:
            change_scale = dataset.do_change_scale.to(configs.device)
            change_scale = change_scale.view(1, 1, 2)

            # Create masks for padding values
            mask1 = pt1_sort == padding
            mask2 = pt2_sort == padding
            mask_pred = pt2_pred_sort == padding
            
            # Scale only non-padding values
            pt1_sort = torch.where(mask1, pt1_sort, pt1_sort / change_scale)
            pt2_sort = torch.where(mask2, pt2_sort, pt2_sort / change_scale)
            pt2_pred_sort = torch.where(mask_pred, pt2_pred_sort, pt2_pred_sort / change_scale)

            
            # pt1_sort, pt2_sort, pt2_pred_sort = pt1_sort / change_scale, pt2_sort / change_scale, pt2_pred_sort /change_scale

            # # but my -1 is changed
            # # pt1_sort[pt1_sort < 0] = padding
            # pt2_sort[pt2_sort < 0] = padding
            # pt2_pred_sort[pt2_pred_sort < 0] = padding
            
        
        t1 = pt1[:,:,0:1]
        t2 = pt1[:,:,0:1] + 1.
        pt1_sort, pt2_sort, pt2_pred_sort = torch.cat([t1, pt1_sort], dim = -1), torch.cat([t2, pt2_sort], dim = -1), torch.cat([t2, pt2_pred_sort], dim = -1)
        write_batches_to_csv(pt1_sort, pt2_sort, pt2_pred_sort, filename = output_file, mode = 'a')
        # writing paired txt (t1, x1, y1), (t2, x2, y2)

import sys
sys.setrecursionlimit(10000)  # Increase the limit

class LinkerGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        
    def add_edge(self, u, v):
        # Add (tk, -1, -1) as a marker for the end of a trajectory
        self.graph[u].append(v)

    def find_trajectories(self):
        visited = set()
        trajectories = []

        def dfs(node, path):
            visited.add(node)
            path.append(node)

            _, x, y = node
            if (x, y) == (-1, -1) or node not in self.graph:  # End of a trajectory      
                if path[-1] != (path[-1][0], -1.0, -1.0): # Remove the (-1, -1) from the final trajectory
                    trajectories.append(path[:])
                elif path[-1] == (path[-1][0], -1.0, -1.0):
                    trajectories.append(path[:-1])

            else:
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, path)

            path.pop()
            # visited.remove(node) #repeated samples

        total_nodes = len(self.graph)
        with tqdm(total=total_nodes, desc="Processing linker the paired points", unit="node") as pbar:
            for node in self.graph:
                if node not in visited and (node[1], node[2]) != (-1, -1):
                # if node not in visited and not (node[1] < 0 or node[2] < 0):
                    dfs(node, [])
                pbar.update(1)

        return trajectories

def get_spt_trackers(folderpath, filename, saved_filename = None, coord_num = 3):
    # open _all_spots_pairs csv to linker by id
    Tracks = np.genfromtxt(os.path.join(folderpath, filename), 
                            delimiter=',', skip_header=1, dtype=float)
    # Sort based on first column
    Tracks = Tracks[Tracks[:, 0].argsort()]
    Tracks = Tracks[:, :2 * coord_num] # first t, x, y tuple, second t, x, y
    g = LinkerGraph()
    for spots in Tracks:
        g.add_edge(tuple(spots[:coord_num]), tuple(spots[coord_num:]))
    trajectories = g.find_trajectories()
    
    
    if saved_filename is None:
        sf = filename.split('.')
        sf[-2] += '_tracker' 
        saved_filename = ('.').join(sf)


    with open(os.path.join(folderpath, saved_filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 't', 'x', 'y'])  # Writing the header
        
        point_id = 0  # Starting ID
        
        for traj in trajectories:
            for point in traj:
                t, x, y = point
                writer.writerow([point_id, t, x, y])
            point_id += 1

    print(f"\nTrajectories have been written to {saved_filename}.")


def optim_link_tracks(filename, track_md = 'lap', eps = 7.5, gap_frame = 1):
    # linking method kalman OR lap
	def get_pos_velocity_from_spots(fname):
		df = pd.read_csv(fname)
		T_total = df.iloc[:, 0].values.astype(int)
		T_sort = [i for i in range(T_total.min(), T_total.max() + 1)]
		VX = df['x2_pred'] - df['x1']
		VY = df['y2_pred'] - df['y1']

		# Create a new DataFrame with the desired columns
		TXV_df = pd.DataFrame({
			'T': df['t1'],
			'X': df['x1'],
			'VX': VX,
			'Y': df['y1'],
			'VY': VY
		})

		return T_sort, TXV_df

	T_sort, TXV_df = get_pos_velocity_from_spots(filename)
	tracker = optimTracker(eps, gap_frame, track_method=track_md)
	for i in tqdm(T_sort, desc=f'Processing {track_md} linker the paired points'):
		centers = TXV_df.loc[TXV_df['T'] == i].values[:,1:]
		if (len(centers) > 0):
			tracker.update_with_frame(centers, i)

	# imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)
	return tracker


def get_spt_optim_trackers(folderpath, filename, saved_filename = None, track_md = 'lap', 
                           link_eps = 7.5, max_frame_gap = 1):
    # using kalman filter or with LAP algorithm
    # open _all_spots_pairs csv to linker by id
    
    tracker = optim_link_tracks(os.path.join(folderpath, filename),
									   track_md = track_md, 
                                       eps = link_eps, gap_frame = max_frame_gap)
    
    
    if saved_filename is None:
        sf = filename.split('.')
        sf[-2] += '_tracker_'  + track_md
        saved_filename = ('.').join(sf)


    with open(os.path.join(folderpath, saved_filename), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['id', 't', 'x', 'y'])
        track_id = 0
        for i in range(len(tracker.tracks)):
            track = tracker.tracks[i]
            # track_id = track.trackId
            # if
            coord_deque = track.trace
            coord_array = np.array(coord_deque)
            if len(coord_array) == 0:
                continue
            t_diff = np.concatenate([np.array([1.]), np.diff(coord_array[:,0])]) - 1
            # t_diff_id = np.where(t_diff >= max_frame_gap)[0]
            # Write data rows for this track
            for j in range(len(coord_array)):
                if t_diff[j] >= max_frame_gap:
                    track_id += 1
                point = coord_array[j,:]
                t = point[0]
                x = point[1]
                y = point[2]
                csv_writer.writerow([track_id, t, x, y])
            track_id += 1
    print(f"\nTrajectories have been written to {saved_filename}.")
