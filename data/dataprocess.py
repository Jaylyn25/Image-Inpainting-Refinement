import random
import torch
import torch.utils.data
from PIL import Image
import cv2
from skimage import exposure
from glob import glob
import numpy as np
import torchvision.transforms as transforms
from data.L0_Smoothing import L0Smoothing
from data.mask_extract import extract_rgb

class DataProcess_I(torch.utils.data.Dataset):
    def __init__(self, gt_root, synth_root, opt, train=True):
        super(DataProcess_I, self).__init__()
        # define transformation
        self.img_transform = transforms.Compose([
            transforms.Resize([opt.fineSize1,opt.fineSize2]),  #square
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize([opt.fineSize1,opt.fineSize2]),  #square
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt

        # path of training dataset
        if train:
            self.gt_paths = sorted(glob('{:s}/*'.format(gt_root), recursive=True))
            self.synth_paths = sorted(glob('{:s}/*'.format(synth_root), recursive=True))
            self.Train=True
        self.N_mask = len(self.synth_paths)
        print(self.N_mask)

    # each image process
    def __getitem__(self, index):
        gt_img = Image.open(self.gt_paths[index])
        synth_img = Image.open(self.synth_paths[index])
        
        ## generate structure image
        st_img = L0Smoothing(self.gt_paths[index], param_lambda=0.05, param_kappa=1.5).run()
        st_img = np.squeeze(st_img)
        st_img = np.clip(st_img, 0, 1)
        st_img = st_img * 255
        st_img = st_img.astype(np.uint8)
        st_img = Image.fromarray(cv2.cvtColor(st_img,cv2.COLOR_BGR2RGB))
        
        ## generate mask
        img_m = np.array(synth_img)
        mask_img = extract_rgb(img_m)
        mask_img = Image.fromarray(mask_img.astype(np.uint8))
        
        gt_img = self.img_transform(gt_img.convert('RGB'))
        synth_img = self.img_transform(synth_img.convert('RGB'))
        st_img = self.img_transform(st_img.convert('RGB'))
        mask_img = self.mask_transform(mask_img.convert('RGB'))
        return gt_img, synth_img, st_img, mask_img

    #length
    def __len__(self):
        return len(self.synth_paths)

class DataProcess_R(torch.utils.data.Dataset):
    def __init__(self, gt, denoise, opt, train=True):
        super(DataProcess_R, self).__init__()
        # define transformation
        self.img_transform = transforms.Compose([
            transforms.Resize([opt.fineSize1,opt.fineSize2]),  #square
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        
        self.Train = False
        self.opt = opt
        self.denoise = denoise
        self.gt = gt

        # path of training dataset
        if train:
            #self.gt_paths = sorted(glob('{:s}/*'.format(gt_root), recursive=True))
            self.Train=True

    # each image process
    def __getitem__(self, index):
        #gt_img = Image.open(self.gt_paths[index]).convert('RGB')
        gt_img = self.gt[index].convert('RGB')
        de_img = self.denoise[index].convert('RGB')
        
        #for channel in range(gt_img.shape[2]):
         #   gt_img_eq[:,:,channel] = exposure.equalize_hist(gt_img[:,:,channel])
         #   de_img_eq[:,:,channel] = exposure.equalize_hist(de_img[:,:,channel]) 

        gt_img = self.img_transform(gt_img)
        de_img = self.img_transform(de_img)
        
        return gt_img, de_img

    #length
    def __len__(self):
        return len(self.gt)

