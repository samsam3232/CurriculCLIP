import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.coco_ds import CurriculumCocoCaptions
from torch.utils.data import DataLoader
from datasets.officecaltech import OfficeCaltech
import os

SENTENCES = ["A photo of a back pack.", "A photo of a bike.", "A photo of a calculator", "A photo of an headphone.",
             "A photo of a keyboard.", "A photo of a computer.", "A photo of a monitor.", "A photo of a mouse.",
             "A photo of a mug.", "A photo of a projector."]

def get_loader(image_path, annotation_path, batch_size, **kwargs):

    trans = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
    ds = CurriculumCocoCaptions(image_path, annotation_path, transform=trans, **kwargs)
    loader = DataLoader(ds, batch_size = batch_size)
    return loader

def test_model(model, tokenizer, computation='cuda', root_path = os.path.join(os.getcwd(), '/datasets/caltech/'),
               temperature = 1., **kwargs):

    with torch.no_grad():
        ds = OfficeCaltech(root=root_path,task="C",download=os.path.exists(root_path),
                           transform=transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()]))
        loader = DataLoader(ds, shuffle=True, batch_size=32)
        tokenized = tokenizer(SENTENCES, max_length=20, padding='max_length', return_tensors='pt')
        tokenized['input_ids'], tokenized['attention_mask'] = tokenized['input_ids'].to(computation), \
                                                              tokenized['attention_mask'].to(computation)
        encoded = model.get_text_encoding(**tokenized)
        print("******** Testing the model ********")
        corrects, total = 0, 0
        for images, targets in loader:
            images, targets = images.to(computation), targets.to(computation)
            image_encoding = model.get_image_encoding(images)
            similarities = F.softmax((image_encoding @ encoded.T) / temperature)
            preds = torch.argmax(similarities, dim=1)
            corrects += torch.sum(preds == targets)
            total += len(targets)

    return corrects / total