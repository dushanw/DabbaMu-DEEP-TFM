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
from modules.psfs import *
from modules.losses import *

from modules.datasets import *
from modules.data_utils import return_dataloaders

from modules.train_utils.reconstruction_swinIR_wforward import train

from modules.models.forward_model import modelA_class
from modules.models.forward_H import modelH_class
from modules.models.decoder import *
from modules.models.decoder_upsampling_nets import *
from modules.models.decoder_upsampling_nets_experimental import *
from modules.models.decoder_support_blocks import conv_bn_block
from modules.m_inc_procs import *

from modules.models.lambdat_yt_skips import *

from adversarial_learning.swinIRwforward_support_files.models_define import swinIR_generative_decoder

def run(config_file=None, opts=None, save_special=False, save_dir_special= None):
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
    num_samples_valtest= cfg.DATASET.num_samples_valtest
    delta=cfg.DATASET.delta
    batch_size_train= cfg.DATASET.batch_size_train
    batch_size_valtest= cfg.DATASET.batch_size_valtest
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
    emPSF= eval(cfg.MODEL.MODEL_A.emPSF)
    
    noise=cfg.MODEL.MODEL_A.noise
    lambda_scale_factor=cfg.MODEL.MODEL_A.lambda_scale_factor # downsample
    rotation_lambda=cfg.MODEL.MODEL_A.rotation_lambda
    shift_lambda_real=cfg.MODEL.MODEL_A.shift_lambda_real
    
    readnoise_std=cfg.MODEL.MODEL_A.readnoise_std

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
    
    trainset, valset, testset = get_dataset_func(img_size= img_size, delta= delta, num_samples_train= num_samples_train, num_samples_valtest= num_samples_valtest)

    
    if cfg.DATASET.name == 'confocal' or cfg.DATASET.name == 'neuronal' or cfg.DATASET.name== 'bbbcHumanMCF7cellsW2' or cfg.DATASET.name== 'bbbcHumanMCF7cellsW4':drop_last_val_test= True ## the last batch of confocal data haas only 1 image, it lead to an error
    
    else:drop_last_val_test= False
        
    train_loader, val_loader, test_loader = return_dataloaders(trainset, valset, testset, batch_size_train= batch_size_train, drop_last_val_test= drop_last_val_test, batch_size_valtest= batch_size_valtest)
    
    ###
    
    torch.manual_seed(torch_seed)
    modelH = modelH_class(T=T, img_size = img_size, preprocess_H_weights= H_weight_preprocess, 
                          device = device, 
                          initialization_bias=initialization_bias, 
                          activation = H_activation, init_method= H_init, 
                          enable_train=enable_train, lambda_scale_factor= lambda_scale_factor).to(device)
    
    modelA= modelA_class(sPSF= sPSF.to(device), exPSF= exPSF.to(device), emPSF= emPSF.to(device), noise=noise, device = device, 
                         scale_factor=lambda_scale_factor, rotation_lambda=rotation_lambda, 
                         shift_lambda_real= shift_lambda_real,
                         readnoise_std= readnoise_std)
    
    decoder_upsample_net= None
    if os.path.isdir('/n/home06/udithhaputhanthri/project_udith/aim2'):
        project_dir= '/n/home06/udithhaputhanthri/project_udith/aim2'
    else:
        project_dir='/home/udith/udith_works/DabbaMu-DEEP-TFM/aim2' # handle lab server

    if cfg.GENERAL.other_opt_dir !=None:
        other_opt_dir= cfg.GENERAL.other_opt_dir
        print(f'** other opt dir is used from configs : {cfg.GENERAL.other_opt_dir}')
    else:
        other_opt_dir= 'adversarial_learning/swinIRwforward_support_files/opt.yaml'
        print(f'** default other opt dir is used : {other_opt_dir}')


    decoder= swinIR_generative_decoder(f'{project_dir}/{other_opt_dir}', cfg, modelA, modelH)
    decoder.init_train()

    opt_H= None
    opt_decoder= None #defined inside the SwinIR decoder model


    if save_dir_special!= None:
        try:shutil.rmtree(save_dir_special)
        except:pass
    
        os.mkdir(save_dir_special)
        with open(f'{save_dir_special}/configs.yaml', 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
        
        
    train(decoder, decoder_upsample_net, modelA, modelH, connect_forward_inverse, criterion, [opt_decoder, opt_H], train_loader, val_loader, device, epochs, show_results_epoch, train_model_iter, train_H_iter, m_inc_proc, save_dir, classifier, rescale_for_classifier, save_special, cfg, save_dir_special)