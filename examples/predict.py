import argparse
import os
import torch
import matplotlib.pyplot as plt
from uvcgan import ROOT_OUTDIR
from uvcgan.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan.config import Args
from uvcgan.cgan import construct_model
from torch.utils.data import DataLoader, Dataset
from uvcgan.data.datasets import CycleGANDataset
import torchvision.transforms as transforms
import numpy as np

def load_gen_ab(model_path):
    
        device = get_torch_device_smart()
        args   = Args.load(model_path)
        model = construct_model(
        args.savedir, args.config, is_train = False, device = device
        )
        epoch = 200
        if epoch == -1:
            epoch = max(model.find_last_checkpoint_epoch(), 0)

        print("Load checkpoint at epoch %s" % epoch)

        seed_everything(args.config.seed)
        model.load(epoch)
        gen_ab = model.models.gen_ab
        gen_ab.eval()
        return gen_ab.cuda()

def making_predictions(train_dataloader,gen_ab):
        gen_ab.eval()
        for _, sample_batched in enumerate(train_dataloader):
            inputs = sample_batched
            inputA, inputB = inputs
            with torch.no_grad():
                features = gen_ab(inputA)
            file = features.detach().cpu().numpy()
            file_save = file.squeeze()
            print(np.array(file_save).shape)
            # plt.imsave(f"<path_to_save_translated_files>", np.array(file_save), cmap='gray')   

if __name__ == '__main__':
  batch_size=32
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Specify the path of your model after trained through UVCGAN.
  model_path = os.path.join(ROOT_OUTDIR, 'outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
  model = load_gen_ab(model_path)
  ds = CycleGANDataset('path_to_your_data',is_train=False,transform = transforms.Compose([transforms.CenterCrop((224,224)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
  dl = DataLoader(ds, batch_size=batch_size,shuffle=False)
  making_predictions(dl,model)
  print("Finish")
