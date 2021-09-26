import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess_I, DataProcess_R
from models.models import create_model_I, create_model_R
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
from PIL import Image

if __name__ == "__main__":

    opt = TrainOptions().parse()
    # define the dataset
    dataset_I = DataProcess_I(opt.gt_root, opt.synth_root, opt, opt.isTrain)
    iterator_train_I = (data.DataLoader(dataset_I, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers))
    # Create model
    # load places2
    model_I = create_model_I(opt)
    #model_I.netEN.module.load_state_dict(torch.load("./pretrained_models/place2/EN.pkl"))
    #model_I.netDE.module.load_state_dict(torch.load("./pretrained_models/place2/DE.pkl"))
    #model_I.netMEDFE.module.load_state_dict(torch.load("./pretrained_models/place2/MEDFE.pkl"))
    # load epoch 120 pre-trained
    #model_I.load_networks_I(60)  

    total_steps = 0
    inpainting_result = []
    ground_truth = []
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    # Start Training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for gt, detail, structure, mask in iterator_train_I:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            model_I.set_input(gt, detail, structure, mask)
            model_I.optimize_parameters()  # <--
            # save fake_out
            if (epoch == (opt.niter + opt.niter_decay)) and (opt.refinement):
                inpainting_result.append(model_I.get_inpainting_result(opt.HE)[0])
                ground_truth.append(model_I.get_inpainting_result(opt.HE)[1])              
            
            # display the training processing
            if total_steps % opt.display_freq == 0:
                input, output, GT = model_I.get_current_visuals()
                image_out = torch.cat([input, output, GT], 0)
                grid = torchvision.utils.make_grid(image_out)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model_I.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                writer.add_scalar('G_GAN', errors['G_GAN'], total_steps + 1)
                writer.add_scalar('G_L1', errors['G_L1'], total_steps + 1)
                writer.add_scalar('G_stde', errors['G_stde'], total_steps + 1)
                writer.add_scalar('D_loss', errors['D'], total_steps + 1)
                writer.add_scalar('F_loss', errors['F'], total_steps + 1)
                print('iteration time: %d' % t)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the inpainting model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model_I.save_networks_I(epoch)
            final_epoch = epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model_I.update_learning_rate()


    ## refinement training
    if opt.refinement:
        #
        opt.fineSize1 = 192 
        opt.name = 'RefineNet'
        opt.input_nc = 3
        opt.kw = 3
        # define the dataset
        dataset_R = DataProcess_R(ground_truth, inpainting_result, opt, opt.isTrain)
        iterator_train_R = (
            data.DataLoader(dataset_R, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers))
        # Create model
        model_R = create_model_R(opt)
        model_R.load_networks_R(120)  # epoch 120 pre-trained

        total_steps = 0
        # Start Training
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0
            for gt, detail in iterator_train_R:
                iter_start_time = time.time()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
             
                model_R.set_input(gt, detail)
                model_R.optimize_parameters()  # <--
                # display the training processing
                if total_steps % opt.display_freq == 0:
                    input, output, GT = model_R.get_current_visuals()
                    image_out = torch.cat([input, output, GT], 0)
                    # image_out = torch.cat([detail, output, gt], 0)
                    grid = torchvision.utils.make_grid(image_out)
                    writer.add_image('R_Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
                # display the training loss
                if total_steps % opt.print_freq == 0:
                    errors = model_R.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    writer.add_scalar('R_G_GAN', errors['G_GAN'], total_steps + 1)
                    writer.add_scalar('R_G_L1', errors['G_L1'], total_steps + 1)
                    # writer.add_scalar('G_stde', errors['G_stde'], total_steps + 1)
                    writer.add_scalar('R_D_loss', errors['D'], total_steps + 1)
                    writer.add_scalar('R_F_loss', errors['F'], total_steps + 1)
                    print('iteration time: %d' % t)
            if epoch % opt.save_epoch_freq == 0:
                print('saving the Refine model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model_R.save_networks_R(epoch)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model_R.update_learning_rate()
    writer.close()
