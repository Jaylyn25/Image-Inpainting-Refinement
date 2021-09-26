# Image-Inpainting-Refinement
## Image Inpainting
Follow the work of [Rethinking-Inpainting-MEDFE](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE#rethinking-inpainting-medfe). 
<br> For Structure image in training inpainting network, we utlize the [L0 smoothing method](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.227.9419&rep=rep1&type=pdf).

## Image Refinement
Follow the work of [Rethinking-Inpainting-MEDFE](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE#rethinking-inpainting-medfe) and [Image-To-Image Translation With Conditional Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html).

## Pretrained models
https://drive.google.com/drive/folders/1y_vaSjTDmgIksYgh6ANlkqDj1MoW7Bz1?usp=sharing

### Training
```
# training inpainting networks
!CUDA_VISIBLE_DEVICES=0 python train_inpainting.py --log_dir your_root --de_root your_root --gt_root your_root
# training refinement networks
!CUDA_VISIBLE_DEVICES=0 python train_refine.py --log_dir your_root --de_root your_root --gt_root your_root
```
### Testing
```
!CUDA_VISIBLE_DEVICES=0 python test.py --synth_root your_root --results_dir your_root 
# optionally add --refinement and --HE
```
