from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = 'Experiment'

_C.GENERAL = CN()
_C.GENERAL.torch_seed= 10
_C.GENERAL.device = 'cuda:0'
_C.GENERAL.save_dir = 'figs/test'

_C.DATASET = CN()
_C.DATASET.name= 'mnistdigits_grid2patch'
_C.DATASET.img_size= 32
_C.DATASET.num_samples_train= 600
_C.DATASET.delta=0.000001
_C.DATASET.batch_size_train= 32
_C.DATASET.img_channels=1

_C.TRAIN = CN()
_C.TRAIN.epochs=150
_C.TRAIN.m_inc_proc =  'inc_m_class(epoch_threshold= 80, epoch_steps= 10)'
_C.TRAIN.show_results_epoch=5
_C.TRAIN.train_model_iter= 1
_C.TRAIN.train_H_iter= 1
_C.TRAIN.criterion= 'nn.L1Loss().to(device)'  #test_loss_for_H
_C.TRAIN.classifier=None
_C.TRAIN.rescale_for_classifier=None


_C.MODEL= CN()
_C.MODEL.MODEL_H = CN()
_C.MODEL.MODEL_H.T= 5
_C.MODEL.MODEL_H.H_weight_preprocess= 'ifft_2d_with_fftshift_real'
_C.MODEL.MODEL_H.H_init = 'randn_FourierBased'
_C.MODEL.MODEL_H.initialization_bias=0
_C.MODEL.MODEL_H.H_activation= 'sigmoid_custom'
_C.MODEL.MODEL_H.lr_H= 0.0

_C.MODEL.MODEL_A = CN()
_C.MODEL.MODEL_A.sPSF= 'torch.tensor(impulse(side_len=5)).float().to(device)'
_C.MODEL.MODEL_A.exPSF= 'torch.tensor(impulse(side_len=5)).float().to(device)'
_C.MODEL.MODEL_A.noise=False
_C.MODEL.MODEL_A.lambda_scale_factor=1 # downsample
_C.MODEL.MODEL_A.rotation_lambda=1000.0
_C.MODEL.MODEL_A.shift_lambda_real=10.0

_C.MODEL.MODEL_DECODER = CN()
_C.MODEL.MODEL_DECODER.name= 'genv1'
_C.MODEL.MODEL_DECODER.upsample_net= 'bicubic_interp' ## define in modules.models.decoder_upsampling_nets
_C.MODEL.MODEL_DECODER.upsample_net_init_method= 'randn'
_C.MODEL.MODEL_DECODER.channel_list=[24, 12, 8, 4, 2]
_C.MODEL.MODEL_DECODER.lr_decoder= 0.001
_C.MODEL.MODEL_DECODER.last_activation='sigmoid'



def get_cfg_defaults():
    return _C.clone()

