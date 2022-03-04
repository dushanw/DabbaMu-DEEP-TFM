
from torch.nn import init
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import yaml

from adversarial_learning.swinIRwCustomUp_support_files.models import *
from adversarial_learning.swinIRwCustomUp_support_files.losses import *


from modules.models.decoder_upsampling_nets import *
from modules.models.decoder_upsampling_nets_experimental import *
from modules.models.decoder_support_blocks import conv_bn_block


def define_G(opt, prev_cfg):
    opt_net = opt['netG']
    netG = SwinIR(upscale=opt_net['upscale'],
                    in_chans=opt_net['in_chans'],
                    img_size=opt_net['img_size'],
                    window_size=opt_net['window_size'],
                    img_range=opt_net['img_range'],
                    depths=opt_net['depths'],
                    embed_dim=opt_net['embed_dim'],
                    num_heads=opt_net['num_heads'],
                    mlp_ratio=opt_net['mlp_ratio'],
                    upsampler=opt_net['upsampler'],
                    resi_connection=opt_net['resi_connection'], prev_cfg= prev_cfg)
    if opt['is_train']:
        init_weights(netG,
                        init_type=opt_net['init_type'],
                        init_bn_type=None, #opt_net['init_bn_type'],
                        gain=None) #opt_net['init_gain'])
    return netG

def define_D(opt):
    opt_net = opt['netD']
    netD = Discriminator_UNet(input_nc=opt_net['in_nc'],
                            ndf=opt_net['base_nc'])
    init_weights(netD,
                    init_type=opt_net['init_type'],
                    init_bn_type=opt_net['init_bn_type'],
                    gain=opt_net['init_gain'])
    return netD


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')

########################### FINAL MODEL

class swinIR_generative_decoder(nn.Module):
    def __init__(self, opt_file_path= 'swinIR_support_files/opt.yaml', prev_cfg= None):
        super(swinIR_generative_decoder, self).__init__()

        with open(opt_file_path, 'r') as file:
            opt = yaml.load(file)
        self.opt = opt                         # opt
        self.device = prev_cfg.GENERAL.device #torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt, prev_cfg)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt, prev_cfg).to(self.device).eval()

        self.upsample_net= self.create_upsample_net(prev_cfg)
    
    def create_upsample_net(self, prev_cfg): #x: (m, T, low_res_img_size, low_res_img_size)
        
        T                              = prev_cfg.MODEL.MODEL_H.T
        lambda_scale_factor            = prev_cfg.MODEL.MODEL_A.lambda_scale_factor 
        img_size                       = prev_cfg.DATASET.img_size
        decoder_upsample_init_method   = prev_cfg.MODEL.MODEL_DECODER.upsample_net_init_method
        custom_upsampling_bias         = prev_cfg.MODEL.MODEL_DECODER.custom_upsampling_bias
        device                         = prev_cfg.GENERAL.device

        upsample_net= custom_v2(lambda_scale_factor= lambda_scale_factor, T= T, recon_img_size= img_size, init_method= decoder_upsample_init_method, Ht= None, custom_upsampling_bias= custom_upsampling_bias, upsample_postproc_block= lambda x:x).to(device)
        
        print('upsample net (custom_v2) is created succesfully inside SwinIR')
        return upsample_net
        
        
    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def model_to_device(self, network):return network.to(self.device)

    def update_E(self, decay=0.999):
        netG = self.netG
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)

    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netD.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def load(self):
        self.update_E(0)
        self.netE.eval()

    def define_loss(self):
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # 2) F_loss
        # ------------------------------------
        if self.opt_train['F_lossfn_weight'] > 0:
            F_feature_layer = self.opt_train['F_feature_layer']
            F_weights = self.opt_train['F_weights']
            F_lossfn_type = self.opt_train['F_lossfn_type']
            F_use_input_norm = self.opt_train['F_use_input_norm']
            F_use_range_norm = self.opt_train['F_use_range_norm']
            if self.opt['dist']:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm).to(self.device)
            else:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm)
                self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
                self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
            self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        else:
            print('Do not use feature loss.')
            self.F_lossfn = None

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        #print(self.L.shape, self.upsample_net(self.L).shape, self.H.shape)
        self.E = self.netG(self.upsample_net(self.L))

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step, other_optimizers=None):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss_G_total = 0

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
            if self.opt_train['G_lossfn_weight'] > 0:
                G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                loss_G_total += G_loss                 # 1) pixel loss
            if self.opt_train['F_lossfn_weight'] > 0:
                F_loss = self.F_lossfn_weight * self.F_lossfn(self.E, self.H)
                loss_G_total += F_loss                 # 2) VGG feature loss

            if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.H).detach()
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * (
                        self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                        self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
            loss_G_total += D_loss                    # 3) GAN loss

            loss_G_total.backward()
            self.G_optimizer.step()
            
            if other_optimizers!=None:other_optimizers.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()
        if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            # real
            pred_d_real = self.netD(self.H)                # 1) real data
            l_d_real = self.D_lossfn(pred_d_real, True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach().clone()) # 2) fake data, detach to avoid BP to G
            l_d_fake = self.D_lossfn(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt_train['gan_type'] == 'ragan':
            # real
            pred_d_fake = self.netD(self.E).detach()       # 1) fake data, detach to avoid BP to G
            pred_d_real = self.netD(self.H)                # 2) real data
            l_d_real = 0.5 * self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach())
            l_d_fake = 0.5 * self.D_lossfn(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
            l_d_fake.backward()

        self.D_optimizer.step()

        # ------------------------------------
        # record log
        # ------------------------------------
        self.log_G_loss = G_loss
        self.log_F_loss = F_loss
        self.log_D_loss = D_loss

        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()