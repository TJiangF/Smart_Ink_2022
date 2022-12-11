import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import scipy.stats as st
import torch.nn as nn
import matplotlib.pyplot as plt
import  torch.nn.functional as F
from models.Hed import Hed


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'G_ink', 'cycle_A_SSIM', 'cycle_B_SSIM']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', ]
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'ink_real_B', 'ink_fake_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # The new generator for ink_wash image
            self.netD_ink = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_ink_B_pool = ImagePool(opt.pool_size)
            self.fake_ink_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # apply SSIM loss func for image comparison
            self.criterionCycleSSIM = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain: #define gaussian blur
            g_kernel = self.gauss_kernel(21, 3, 1).transpose((3, 2, 1, 0))
            self.gauss_conv = nn.Conv2d(1, 1, kernel_size=21, stride=1, padding='same', padding_mode='replicate', bias=False)
            # use Gaussian kernel for blurring. The weight does not need to be updated.
            self.gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
            self.gauss_conv.weight.requires_grad = False
            if torch.cuda.is_available():
                self.gauss_conv.cuda()

            # ~~~~~~
        self.hed_model = Hed()
        if torch.cuda.is_available():
            self.hed_model.cuda()
        save_path = './hed_pre_trained_model.pth'
        self.hed_model.load_state_dict(torch.load(save_path))
        for param in self.hed_model.parameters():
            param.requires_grad = False
            # ~~~~~~


    """
    generate a gauss kernel for ink wash blurring.
    """
    def gauss_kernel(self, kernel_size = 21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernel_size)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernel_size + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernel_size, kernel_size, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_ink_wash(self, img):
        p1d = (3, 3, 3, 3)
        padding_img = F.pad(img, p1d, "constant", 1)
        # pooling size (7,7), with padding(3,3) to get the same size as input
        erode_img = -1 * (F.max_pool2d(-1 * padding_img, 7, 1))
        channel1 = self.gauss_conv(erode_img[:, 0, :, :].unsqueeze(1))
        channel2 = self.gauss_conv(erode_img[:, 1, :, :].unsqueeze(1))
        channel3 = self.gauss_conv(erode_img[:, 2, :, :].unsqueeze(1))
        return torch.cat((channel1, channel2, channel3), dim = 1)

    @staticmethod
    def cross_entropy(sig_logits, label):
        # print(sig_logits)
        count_neg = torch.sum(1. - label)
        count_pos = torch.sum(label)

        beta = count_neg / (count_pos + count_neg)
        pos_weight = beta / (1 - beta)

        cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1 - label) * (-1) * torch.log(1 - sig_logits)
        cost = torch.mean(cost * (1 - beta))

        return cost

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        self.ink_real_B = self.get_ink_wash(self.real_B)
        self.ink_fake_B = self.get_ink_wash(self.fake_B)
        self.edge_real_A = torch.sigmoid(self.hed_model(self.real_A.detach()))
        self.edge_fake_B = torch.sigmoid(self.hed_model(self.fake_B))



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_ink(self):
        ink_fake_B = self.fake_ink_B_pool.query(self.ink_fake_B)
        self.loss_D_ink = self.backward_D_basic(self.netD_ink, self.ink_real_B, self.ink_fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_ink = self.opt.lambda_ink_wash
        lambda_edge = self.opt.lambda_edge
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # ink wash loss
        self.loss_G_ink = self.criterionGAN(self.netD_ink(self.ink_fake_B), True) * lambda_ink
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_A_SSIM = (1 - self.criterionCycleSSIM(self.rec_A, self.real_A)) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_cycle_B_SSIM = (1 - self.criterionCycleSSIM(self.rec_B, self.real_B)) * lambda_B
        # calculate edge loss
        self.loss_G_edge = self.cross_entropy(self.edge_fake_B, self.edge_real_A) * lambda_A
        # combined loss and calculate gradients
        if self.opt.use_SSIM:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A_SSIM + self.loss_cycle_B_SSIM + self.loss_idt_A + self.loss_idt_B + self.loss_G_ink + self.loss_G_edge
        else:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_ink + self.loss_G_edge

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_ink], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_ink()    # calculate gradients for D_ink
        self.optimizer_D.step()  # update D_A and D_B's weights

    """
    display an img and print current image shape
    Matplotlib will automatically turn a normalized image [0,1] to a regular image [0,255] 
    """
    def check_img(self, img):
        img = np.array(img)
        print('current shape', img.shape)
        img = img.squeeze()
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        plt.figure("Test Image Sample")
        plt.imshow(img)
        plt.show()

