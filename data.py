from turtle import back
import torch
import torchvision
from typing import Collection
from torchvision import transforms as T
from skimage import transform
from torch.utils.data import Dataset, DataLoader


def load_pretrained_backbone(network_backbone, weights_location, device="cuda:0",
                             final_layer_weights: str = "fc.weight"):
    '''
    Load and instantiate a pretrained backbone from a given file containing a state_dict

    Parameters:
    -----------
    network_backbone: a non-instantiated torch.nn.Module, the backbone for obtaining the features
    weights_location: str, the location of the weights file
    device: str, the device to load the weights to
    final_layer_weights: str, the name of the final layer weights to load, used to set the number of classes of the final network

    Returns:
    -----------
    a nn.Module, the instantiated backbone on the given device with the pretrained weights
    '''
    backbone_weights = torch.load(weights_location)
    backbone = network_backbone(num_classes=backbone_weights[final_layer_weights].shape[0])
    backbone.load_state_dict(backbone_weights)
    return backbone.to(device)


class BasicDataset(Dataset):
    '''
    A simple dataset wrapping a container of images without labels.
    '''

    def __init__(self, data, transform=None):
        '''
        Parameters:
        -----------
        data: a generic container of tensors
        transform: a pipeline of transformations to apply to the data
        '''
        self.data = data
        self.current_set_len = data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curdata = self.data[idx]
        if self.transform is not None:
            return transform(curdata)
        return curdata


def get_bare_transforms(size:int=256):
    '''
    Returns a bare minimum transform for the dataset of punches.
    This includes a resizing to a common size (default=256x256 px) and a normalization.
    Parameters
    ----------
    size: an integer indicating the size of the resized images.
    Returns
    -------
    A pipeline of torchvision transforms.
    '''
    return torchvision.transforms.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def get_dataset(root, transforms=None) -> torch.utils.data.Dataset:
    return torchvision.datasets.ImageFolder(root, transform=transforms)

def get_dataloader(root, batch_size:int=32, num_workers:int=4, transforms=None, shuffle=True) -> torch.utils.data.DataLoader:
    dataset = get_dataset(root, transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


class FeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data.shape[0]        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.data[idx]        
        return curdata


def get_hidden_features(dataloader, device, backbone=None):
    features = []
    def get_features(module, input_, output):
        features.append(output.cpu().detach())

    handle = backbone.layer4.register_forward_hook(get_features)
    backbone.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            print(f"Done {i+1}/{len(dataloader)}")
            _ = backbone(data.to(device))

    return torch.cat(features, dim=0), backbone

def create_backbone(name_modelpth, model, device):
    backbone_weights = torch.load(name_modelpth, map_location='cpu')
    backbone_weights.keys()
    backbone_weights["fc.weight"].shape
      
    backbone_class = getattr(torchvision.models, model)
    backbone = backbone_class(num_classes=backbone_weights["fc.weight"].shape[0])
    backbone.load_state_dict(backbone_weights)
    backbone = backbone.to(device).eval()

    return backbone

    