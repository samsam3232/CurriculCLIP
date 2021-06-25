import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

RESNETS = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}

SENTENCES = ["A photo of a back pack.", "A photo of a bike.", "A photo of a calculator", "A photo of an headphone.",
             "A photo of a keyboard.", "A photo of a computer.", "A photo of a monitor.", "A photo of a mouse.",
             "A photo of a mug.", "A photo of a projector."]

def get_resnets(size = 18, pretrained=False):

    return RESNETS[size](pretrained= pretrained)