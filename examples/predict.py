import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from uvcgan.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan.config import Args
from uvcgan.cgan import construct_model
from uvcgan.data.datasets.cyclegan import CycleGANDataset

def load_gen_ab(checkpoint_epoch):
    args   = Args.load(model_path)
    model = construct_model(
    args.savedir, args.config, is_train = False, device = device
    )
    if checkpoint_epoch == -1:
        checkpoint_epoch = max(model.find_last_checkpoint_epoch(), 0)

    print("Load checkpoint at epoch %s" % checkpoint_epoch)

    seed_everything(args.config.seed)
    model.load(checkpoint_epoch)
    gen_ab = model.models.gen_ab
    return gen_ab

def making_predictions(train_dataloader,gen_ab):
    gen_ab.eval()
    for (inputA, _) in train_dataloader:
        with torch.no_grad():
            features = gen_ab(inputA)
        file = features.detach().cpu().numpy()
        file_save = file.squeeze()
        plt.imsave(
            "<path_to_save_translated_files>",
            np.array(file_save),
            cmap='gray')

def create_data_loader():
    # Change or add transformations as per your needs
    transformations = [
        transforms.CenterCrop((224,224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    ds = CycleGANDataset(
        "<path_to_your_data>",
        is_train=False,
        transform = transforms.Compose(transformations))
    dl = DataLoader(ds, batch_size=batch_size,shuffle=False)
    return dl

if __name__ == '__main__':
    batch_size=32
    epoch = 200
    device = get_torch_device_smart()
    # Specify the path of your model trained through UVCGAN.
    model_path = "<path_to_saved_model>"
    trained_model = load_gen_ab(epoch)
    dataloader = create_data_loader()
    making_predictions(dataloader,trained_model)
    print("Finish")
