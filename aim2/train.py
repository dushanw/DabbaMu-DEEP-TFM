import torch
import glob
from torch import nn
import shutil
import os
from contextlib import redirect_stdout
from defaults import get_cfg_defaults

from modules.models.preprocess_H_weights import * #ifft_2d_with_fftshift_real
from modules.custom_activations import sigmoid_custom
from modules.kernels import *

from modules.datasets import mnistdigits, mnistdigits_grid2patch
from modules.data_utils import return_dataloaders

from modules.train_utils import train

from modules.models.forward_model import modelA_class
from modules.models.forward_H import modelH_class
from modules.models.decoder import *
from modules.models.decoder_upsampling_nets import *
from modules.models.decoder_upsampling_nets_experimental import *
from modules.models.decoder_support_blocks import conv_bn_block
from modules.m_inc_procs import *
from modules.datasets import *

from modules.models.lambdat_yt_skips import *


def run(config_file=None, opts=None, save_special=False):
    cfg = get_cfg_defaults()
    
    if config_file!=None:
        cfg.merge_from_file(config_file)
        print(f'load config file : {config_file}')
    if opts!=None:
        print('Overide opts : ', opts)
        cfg.merge_from_list(opts)
    cfg.freeze()    
    
    print(cfg)
    
    
    ##################################################################################3
    #general params
    torch_seed= cfg.GENERAL.torch_seed
    device = cfg.GENERAL.device
    save_dir= cfg.GENERAL.save_dir

    #dataset params
    get_dataset_func= eval(cfg.DATASET.name)
    img_size= cfg.DATASET.img_size
    num_samples_train= cfg.DATASET.num_samples_train
    delta=cfg.DATASET.delta
    batch_size_train= cfg.DATASET.batch_size_train
    img_channels= cfg.DATASET.img_channels

    # train params:
    epochs=cfg.TRAIN.epochs
    m_inc_proc =  eval(cfg.TRAIN.m_inc_proc)
    show_results_epoch= cfg.TRAIN.show_results_epoch
    train_model_iter= cfg.TRAIN.train_model_iter
    train_H_iter= cfg.TRAIN.train_H_iter
    criterion= eval(cfg.TRAIN.criterion) # defined below after defining models
    classifier=cfg.TRAIN.classifier
    rescale_for_classifier=cfg.TRAIN.rescale_for_classifier

    ## params to H:
    T= cfg.MODEL.MODEL_H.T
    H_weight_preprocess= eval(cfg.MODEL.MODEL_H.H_weight_preprocess)
    H_init = cfg.MODEL.MODEL_H.H_init
    initialization_bias= cfg.MODEL.MODEL_H.initialization_bias
    H_activation= eval(cfg.MODEL.MODEL_H.H_activation)
    lr_H= cfg.MODEL.MODEL_H.lr_H

    ## params to A
    sPSF= eval(cfg.MODEL.MODEL_A.sPSF)
    exPSF= eval(cfg.MODEL.MODEL_A.exPSF)
    noise=cfg.MODEL.MODEL_A.noise
    scale_factor=cfg.MODEL.MODEL_A.lambda_scale_factor # downsample
    rotation_lambda=cfg.MODEL.MODEL_A.rotation_lambda
    shift_lambda_real=cfg.MODEL.MODEL_A.shift_lambda_real

    ## decoder params:
    decoder_name= eval(cfg.MODEL.MODEL_DECODER.name)
    upsampling_net_name = eval(cfg.MODEL.MODEL_DECODER.upsample_net)
    custom_upsampling_bias= cfg.MODEL.MODEL_DECODER.custom_upsampling_bias
    decoder_upsample_init_method= cfg.MODEL.MODEL_DECODER.upsample_net_init_method
    channel_list=cfg.MODEL.MODEL_DECODER.channel_list
    lr_decoder= cfg.MODEL.MODEL_DECODER.lr_decoder
    last_activation=cfg.MODEL.MODEL_DECODER.last_activation #'sigmoid'
    
    connect_forward_inverse= eval(cfg.MODEL.MODEL_DECODER.connect_forward_inverse)
    print(f'skip connection between FORWARD and INVERSE models :: {cfg.MODEL.MODEL_DECODER.connect_forward_inverse}')
    ########################################################################
    
    if lr_H==0:enable_train= False
    else:enable_train=True
        
    print(f'MODEL_H : enable_train ::: {enable_train} (derived from lr_H)')
    
    try:shutil.rmtree(save_dir)
    except:pass
    
    save_folder_name= save_dir.split('/')[-1]
    print(f'len(results_saving_folder) : {len(save_folder_name)} (<= 255)')
    
    os.mkdir(save_dir)
    
    with open(f"{save_dir}/details.txt", 'w') as f:
        f.write("details\n")
        
    with open(f'{save_dir}/configs.yaml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    ########################################################################
    
    trainset, valset, testset = get_dataset_func(img_size= img_size, delta= delta, num_samples_train= num_samples_train)
    train_loader, val_loader, test_loader = return_dataloaders(trainset, valset, testset, batch_size_train= batch_size_train)
    
    ###
    
    torch.manual_seed(torch_seed)
    modelH = modelH_class(T=T, img_size = img_size, preprocess_H_weights= H_weight_preprocess, 
                          device = device, 
                          initialization_bias=initialization_bias, 
                          activation = H_activation, init_method= H_init, 
                          enable_train=enable_train, lambda_scale_factor= scale_factor).to(device)
    
    modelA= modelA_class(sPSF= sPSF, exPSF= exPSF, noise=noise, device = device, 
                         scale_factor=scale_factor, rotation_lambda=rotation_lambda, 
                         shift_lambda_real= shift_lambda_real)
    
    upsample_postproc_block= nn.Sequential(conv_bn_block(in_channels= 1, out_channels= T//2, kernel_size= 3, padding= 1, stride=1),
                                           conv_bn_block(in_channels= T//2, out_channels= T, kernel_size= 3, padding= 1, stride=1))
        
    decoder_upsample_net= upsampling_net_name(lambda_scale_factor= scale_factor, T= T, recon_img_size= img_size, init_method= decoder_upsample_init_method, Ht= modelH(m=1).detach(), custom_upsampling_bias= custom_upsampling_bias, upsample_postproc_block= upsample_postproc_block).to(device)
    decoder= decoder_name(T, img_size, img_channels, channel_list, last_activation).to(device)

    
    ###
    #def test_loss_for_H(_,__):return torch.abs(modelH()-torch.ones_like(modelH())).sum()
    #criterion= eval(cfg.TRAIN.criterion)
    ###
    
    opt_H= torch.optim.Adam(modelH.parameters(), lr= lr_H)
    opt_decoder= torch.optim.Adam([{'params': decoder.parameters()}, {'params': decoder_upsample_net.parameters()}], lr= lr_decoder)
    ###

    train(decoder, decoder_upsample_net, modelA, modelH, connect_forward_inverse, criterion, [opt_decoder, opt_H], train_loader, val_loader, device, epochs, show_results_epoch, train_model_iter, train_H_iter, m_inc_proc, save_dir, classifier, rescale_for_classifier, save_special)