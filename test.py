import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess_I
from data.dataprocess import DataProcess_R
from models.models import create_model_R, create_model_I
import torchvision
from torch.utils import data
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image, ImageEnhance
from skimage import io, data, exposure
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from data.mask_extract import extract_rgb


if __name__ == "__main__":
    opt = TestOptions().parse()
    # define transforms for images and masks
    img_transform = transforms.Compose([
        transforms.Resize([opt.fineSize1,opt.fineSize2]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_de_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_out_transform = transforms.Compose([
        transforms.Resize([180, 320]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize([opt.fineSize1,opt.fineSize2]),
        transforms.ToTensor()
    ])

    model_I = create_model_I(opt)
    if opt.refinement:
        opt.input_nc = 3
        model_R = create_model_R(opt)
    
    # loading weights
    model_I.load_networks_I(120) # epoch 12 pre-trained
    if opt.refinement:
        model_R.load_networks_R(90) # epoch 12 pre-trained
    
    results_dir = r'./result'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # paths of input
    synth_paths = sorted(glob('{:s}/*'.format(opt.synth_root)))
    image_len = len(synth_paths)

    for i in tqdm(range(image_len)):  # for each item
        path_synth = synth_paths[i]
        de = Image.open(path_synth).convert("RGB")
        synth = Image.open(path_synth).convert("RGB")
        #de.save(rf"{'./data/input'}/{i}.png")

        ## generate mask
        img_m = np.array(de)
        mask = extract_rgb(img_m)
        mask = Image.fromarray(mask.astype(np.uint8))
        #mask.save(rf"{'./data/mask'}/{i}.png")
        ##
        
        mask = mask_transform(mask)
        de = img_de_transform(synth)
        synth = img_transform(synth)
        mask = torch.unsqueeze(mask, 0)  # add a new dimension 4
        de = torch.unsqueeze(de, 0)
        synth = torch.unsqueeze(synth, 0)

        # inpainting
        with torch.no_grad():
            model_I.set_input(synth, synth, synth, mask)  #input_DE
            model_I.forward()
            fake_out = model_I.fake_out
            fake_out = fake_out.detach().cpu() * mask + synth*(1-mask) # 
            fake_image = (fake_out+1)/2.0
        fake_image = F.interpolate(fake_image, size=(180,320), mode='bilinear')
        mask = (mask+1)/2.0
        de = (de+1)/2.0
        mask = F.interpolate(mask, size=(180,320), mode='bilinear')
        fake_image = fake_image.detach().cpu() * mask + de*(1-mask) #
        output = fake_image.detach().cpu().numpy()[0].transpose((1, 2, 0))*255

        # histogram equalization
        if opt.HE:
            output = exposure.equalize_hist(output)
            output = Image.fromarray((output * 255).astype('uint8'))
        else:
            output = Image.fromarray(output.astype(np.uint8))

        # refinement
        if opt.refinement:
            opt.fineSize1 = 192
            synth = img_transform(output)
            synth = torch.unsqueeze(synth, 0)
            with torch.no_grad():
                model_R.set_input(synth, synth)  #input_DE
                model_R.forward()
                fake_out = model_R.fake_out
                fake_image = (fake_out+1)/2.0
            output = fake_image.detach().cpu().numpy()[0].transpose((1, 2, 0))*255
            output = Image.fromarray(output.astype(np.uint8))
            output = img_out_transform(output) # resize

        if i<10:
            a = '000'
        if i>9 and i<100:
            a = '00'
        if i>99 and i<1000:
            a = '0'
        if i>999:
            a = ''
        if opt.enhancement:
            enh_sha = ImageEnhance.Sharpness(output)
            output = enh_sha.enhance(opt.sharp_factor)
        #output.save(rf"{'./data/output'}/{a}{i}.png")
        output.save(rf"{opt.results_dir}/{a}{i}.png")
