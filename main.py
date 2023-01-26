from __future__ import print_function, division
import argparse
import architecture

import os, random, time, copy

from data import *
import numpy as np
from scipy import misc
from scipy import ndimage, signal
import pickle
import sys
import math
import PIL.Image
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

from utils.eval_funcs import *
from utils.dataset_tinyimagenet import *
from utils.plot import *

from train import train_model
from test import test_model, evalutate_data

import warnings  # ignore warnings

warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manualSeed", type=int, default=999)
    parser.add_argument("--manualSeedTorch", type=int, default=0)
    parser.add_argument("--exp_dir", type=str, default='./exp',
                        help="experiment directory, used for reading the init model")
    parser.add_argument("--modelFlag", type=str, default='Res50sc')
    parser.add_argument("--project_name", type=str, default='OpenGanFull')
    parser.add_argument("--device", type=str, default=None,
                        help="device to use (default: None -> use CUDA if available).")
    parser.add_argument("--epochs", type=int, default=900, help="number of epochs to train (default: 900).")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training (default: 128).")
    parser.add_argument("--batch_size_eval", type=int, default=64, help="batch size for eval (default: 64).")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001).")
    parser.add_argument("--path_to_feats", type=str, default='./feats',
                        help="the path to cached off-the-shelf features")
    parser.add_argument("--root_data", type=str, default='./PunchesDataset',
                        help="the path to find the dir of data")
    parser.add_argument("--root_nopunch", type=str, default='./PunchesDataset',
                        help="the path to find the dir of no punch data")
    parser.add_argument("--root_extra", type=str, default='./PunchesDataset',
                        help="the path to find the dir of extra data")
    parser.add_argument("--name_modelpth", type=str, default='./modelRes18.pth',
                        help="pretrained model (default: ResNet18)")
    parser.add_argument("--model", type=str, default='resnet18',
                        help="model to use (default: resnet18)")


    #For GAN-fea, we set the hyper-parameters as below.
    parser.add_argument("--nc", type=int, default=512,
                        help="Number of channels in the training images. For color images this is 3")
    parser.add_argument("--nz", type=int, default=100,
                        help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Size of feature maps in discriminator")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparam for Adam optimizers")
    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs available. Use 0 for CPU mode.")

    parser.add_argument("--nClassTotal", type=int, default=200, help="Number of classes in the dataset")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeedTorch)

    nClassCloseset = args.nClassTotal

    # project_name += '_K{}run{}'.format(nClassCloseset, runIdx)
    if not os.path.exists(args.exp_dir): os.makedirs(args.exp_dir)

    num_epochs = args.epochs
    torch.cuda.device_count()
    torch.cuda.empty_cache()

    save_dir = os.path.join(args.exp_dir, args.project_name)
    print("Save directory:", save_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    log_filename = os.path.join(save_dir, 'train.log')

    #Initialize the network
    netG = architecture.Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(args.device)
    netD = architecture.DiscriminatorFunnel(nc=args.nc, ndf=args.ndf).to(args.device)

    # Handle multi-gpu if desired
    if ('cuda' == args.device) and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(architecture.weights_init)

    if ('cuda' == args.device) and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netG.apply(architecture.weights_init)

    print(args.device)

    trainloader = get_dataloader(os.path.join(args.root_data + "/Train"), args.batch_size, transforms=get_bare_transforms())
    testloader = get_dataloader(os.path.join(args.root_data + "/Test"), args.batch_size_eval,transforms=get_bare_transforms())
    nopunchloader = get_dataloader(os.path.join(args.root_nopunch + "/Crops"), args.batch_size_eval, transforms=get_bare_transforms())
    extraloader = get_dataloader(os.path.join(args.root_extra + "/OOD_train"), args.batch_size_eval, transforms=get_bare_transforms())


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=args.device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr / 5, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    backbone = create_backbone(args.name_modelpth, args.model, device)

    train_features, backbone = get_hidden_features(trainloader, device, backbone)

    print("Start Training...")
    trainset_closeset = FeatDataset(data=train_features)
    featureloader_train = DataLoader(trainset_closeset, batch_size=args.batch_size_eval, shuffle=True, num_workers=1)

    G_losses, D_losses = train_model(num_epochs, featureloader_train, netG, netD, real_label, fake_label, optimizerG, optimizerD, args.nz, fixed_noise, criterion, device, save_dir)

    plot_losses(G_losses, D_losses, args.modelFlag)

    print("Start Testing...")
    test_features, backbone = get_hidden_features(testloader, device, backbone)
    torch.save(test_features, "punzoni_res18_features_TEST.pt")
    featureloader_test = FeatDataset(data=test_features)
    features_testloader = DataLoader(featureloader_test, batch_size=args.batch_size_eval, shuffle=True, num_workers=1)
    ## Anziché il backbone, andrebbe passato il feature_loader
    outputs_open, outputs_close = test_model(backbone, features_testloader, netD, device)
    plot_roc_curve(outputs_open, outputs_close, args.modelFlag)
    plot_hist(outputs_open, outputs_close, args.modelFlag)


    print("Start Testing for no punch features...")
    no_punch_features, backbone = get_hidden_features(nopunchloader, device, backbone)
    featureloader_nopunch = FeatDataset(data=no_punch_features)
    features_nopunchloader = DataLoader(featureloader_nopunch, batch_size=args.batch_size_eval, shuffle=True, num_workers=1)
    outputs_nopunz, _ = evalutate_data(netD, features_nopunchloader, device)
    netD.train()
    outputs_nopunz = outputs_nopunz.detach().cpu().numpy()
    plot_roc_curve(outputs_nopunz, outputs_close, args.modelFlag + 'no_punz')
    plot_hist(outputs_nopunz, outputs_close, args.modelFlag + 'no_punz')


    print("Start Testing for extra features...")
    extra_features, backbone = get_hidden_features(extraloader, device, backbone)
    netD.train()
    featureloader_extra = FeatDataset(data=extra_features)
    features_extraloader = DataLoader(featureloader_extra, batch_size=args.batch_size_eval, shuffle=True, num_workers=1)
    extra_features, _ = evalutate_data(netD, features_extraloader, device)
    outputs_extra = extra_features.detach().cpu().numpy()
    plot_roc_curve(outputs_extra, outputs_close, args.modelFlag + 'extra')
    plot_hist(outputs_extra, outputs_close, args.modelFlag + 'extra')
    
    print("Finish")

if __name__ == "__main__":
    main()