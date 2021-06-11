import torch
import glob
from torch import nn
import shutil
import os
from contextlib import redirect_stdout
from defaults import get_cfg_defaults

from modules.models.preprocess_H_weights import ifft_2d_with_fftshift_real
from modules.custom_activations import sigmoid_custom
from modules.kernels import get_gaussian
from modules.data import mnistgrid_getdataset
from modules.train_utils import train

from modules.models.forward_model import modelA_class
from modules.models.forward_H import modelH_class
from modules.models.decoder import genv1


def inc_1_after_60_interval_10(m, epoch):
    if epoch>60 and epoch%10==0:
        m=inc_m(m, epoch, 1)
    return m

def run(config_file=None, opts=None):
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
    img_size= cfg.DATASET.img_size
    delta=cfg.DATASET.delta
    batch_size= cfg.DATASET.batch_size
    img_channels= cfg.DATASET.img_channels

    # train params:
    epochs=cfg.TRAIN.epochs
    m_inc_proc =  eval(cfg.TRAIN.m_inc_proc)
    show_results_epoch= cfg.TRAIN.show_results_epoch
    train_model_iter= cfg.TRAIN.train_model_iter
    train_H_iter= cfg.TRAIN.train_H_iter
    criterion= eval(cfg.TRAIN.criterion)
    classifier=cfg.TRAIN.classifier
    rescale_for_classifier=cfg.TRAIN.rescale_for_classifier

    ## params to H:
    T= cfg.MODEL.MODEL_H.T
    H_weight_preprocess= eval(cfg.MODEL.MODEL_H.H_weight_preprocess)
    H_init = cfg.MODEL.MODEL_H.H_init
    H_complex_init= cfg.MODEL.MODEL_H.H_complex_init #override by H_init
    initialization_bias= cfg.MODEL.MODEL_H.initialization_bias
    H_activation= eval(cfg.MODEL.MODEL_H.H_activation)
    lr_H= cfg.MODEL.MODEL_H.lr_H
    enable_train=cfg.MODEL.MODEL_H.enable_train

    ## params to A
    sPSF= eval(cfg.MODEL.MODEL_A.sPSF)
    exPSF= eval(cfg.MODEL.MODEL_A.exPSF)
    noise=cfg.MODEL.MODEL_A.noise
    scale_factor=cfg.MODEL.MODEL_A.scale_factor # downsample
    rotation_lambda=cfg.MODEL.MODEL_A.rotation_lambda
    shift_lambda_real=cfg.MODEL.MODEL_A.shift_lambda_real

    ## decoder params:
    channel_list=cfg.MODEL.MODEL_DECODER.channel_list
    lr_decoder= cfg.MODEL.MODEL_DECODER.lr_decoder
    last_activation=cfg.MODEL.MODEL_DECODER.last_activation #'sigmoid'
    ########################################################################

    try:shutil.rmtree(save_dir)
    except:pass
    os.mkdir(save_dir)
    
    with open(f'{save_dir}/configs.yaml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    ########################################################################


    train_loader = torch.utils.data.DataLoader(mnistgrid_getdataset(img_size, 'train', delta), batch_size=batch_size, shuffle=True, drop_last= True)
    val_loader = torch.utils.data.DataLoader(mnistgrid_getdataset(img_size, 'val', delta), batch_size=batch_size, shuffle=True, drop_last= True)
    test_loader = torch.utils.data.DataLoader(mnistgrid_getdataset(img_size, 'test', delta), batch_size=batch_size, shuffle=True, drop_last= True)

    x, y= next(iter(train_loader))
    vmin= x.min().item()
    vmax= x.max().item()
    print('dataset value range : ',vmin, vmax)

    torch.manual_seed(torch_seed)

    ###
    modelH = modelH_class(T=T, img_size = img_size, preprocess_H_weights= H_weight_preprocess, complex_init=H_complex_init, device = device, initialization_bias=initialization_bias, activation = H_activation, init_method= H_init, enable_train=enable_train).to(device)
    modelA= modelA_class(sPSF= sPSF, exPSF= exPSF, noise=noise, device = device, scale_factor=scale_factor, rotation_lambda=rotation_lambda, shift_lambda_real= shift_lambda_real)
    decoder= genv1(T, img_size, img_channels, channel_list, last_activation).to(device)

    opt_H= torch.optim.Adam(modelH.parameters(), lr= lr_H)
    opt_decoder= torch.optim.Adam(decoder.parameters(), lr= lr_decoder)
    ###

    train(decoder, modelA, modelH, criterion, [opt_decoder, opt_H], train_loader, val_loader, device, epochs, show_results_epoch, train_model_iter, train_H_iter, m_inc_proc, save_dir, classifier, rescale_for_classifier)