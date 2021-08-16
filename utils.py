import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.datasets import CurriculumCocoCaptions
from torch.utils.data import DataLoader
from data.datasets import OfficeCaltech
import os
import torchvision


SENTENCES = {'coco': ["A photo of a back pack.", "A photo of a bike.", "A photo of a calculator", "A photo of an headphone.",
             "A photo of a keyboard.", "A photo of a computer.", "A photo of a monitor.", "A photo of a mouse.",
             "A photo of a mug.", "A photo of a projector."],
             "cifar": ['A photo of an apple.', 'A photo of an aquarium_fish.', 'A photo of a baby.', 'A photo of a bear.',
                          'A photo of a beaver.', 'A photo of a bed.', 'A photo of a bee.', 'A photo of a beetle.',
                          'A photo of a bicycle.', 'A photo of a bottle.', 'A photo of a bowl.', 'A photo of a boy.',
                          'A photo of a bridge.', 'A photo of a bus.', 'A photo of a butterfly.', 'A photo of a camel.',
                          'A photo of a can.', 'A photo of a castle.', 'A photo of a caterpillar.', 'A photo of a cattle.',
                          'A photo of a chair.', 'A photo of a chimpanzee.', 'A photo of a clock.', 'A photo of a cloud.',
                          'A photo of a cockroach.', 'A photo of a couch.', 'A photo of a crab.', 'A photo of a crocodile.',
                          'A photo of a cup.', 'A photo of a dinosaur.', 'A photo of a dolphin.', 'A photo of an elephant.',
                          'A photo of a flatfish.', 'A photo of a forest.', 'A photo of a fox.', 'A photo of a girl.',
                          'A photo of a hamster.', 'A photo of a house.', 'A photo of a kangaroo.', 'A photo of a computer keyboard.',
                          'A photo of a lamp.', 'A photo of a lawn_mower.', 'A photo of a leopard.', 'A photo of a lion.',
                          'A photo of a lizard.', 'A photo of a lobster.', 'A photo of a man.', 'A photo of a maple_tree.',
                          'A photo of a motorcycle.', 'A photo of a mountain.', 'A photo of a mouse.', 'A photo of a mushroom.',
                          'A photo of an oak_tree.', 'A photo of an orange.', 'A photo of an orchid.', 'A photo of an otter.',
                          'A photo of a palm_tree.', 'A photo of a pear.', 'A photo of a pickup_truck.', 'A photo of a pine_tree.',
                          'A photo of a plain.', 'A photo of a plate.', 'A photo of a poppy.', 'A photo of a porcupine.',
                          'A photo of a possum.', 'A photo of a rabbit.', 'A photo of a raccoon.', 'A photo of a ray.',
                          'A photo of a road.', 'A photo of a rocket.', 'A photo of a rose.', 'A photo of a sea.',
                          'A photo of a seal.', 'A photo of a shark.', 'A photo of a shrew.', 'A photo of a skunk.',
                          'A photo of a skyscraper.', 'A photo of a snail.', 'A photo of a snake.', 'A photo of a spider.',
                          'A photo of a squirrel.', 'A photo of a streetcar.', 'A photo of a sunflower.', 'A photo of a sweet_pepper.',
                          'A photo of a table.', 'A photo of a tank.', 'A photo of a telephone.',
                          'A photo of a television.', 'A photo of a tiger.', 'A photo of a tractor.', 'A photo of a train.',
                          'A photo of a trout.', 'A photo of a tulip.', 'A photo of a turtle.', 'A photo of a wardrobe.',
                          'A photo of a whale.', 'A photo of a willow_tree.', 'A photo of a wolf.', 'A photo of a woman.',
                          'A photo of a worm.']}



def get_loader(image_path, annotation_path, batch_size, **kwargs):

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711])])
    ds = CurriculumCocoCaptions(image_path, annotation_path, transform=trans, **kwargs)
    loader = DataLoader(ds, batch_size = batch_size)
    return loader

def test_model(model, processor, computation='cuda', root_path = os.path.join(os.getcwd(), '/datasets/'),
               temperature = 1., eval_ds = 'cifar', **kwargs):

    with torch.no_grad():

        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])

        if eval_ds == 'cifar':
            ds = torchvision.datasets.CIFAR100(os.path.join(root_path, 'cifar100'), train=False, download=True, transform=trans)
        else:
            ds = OfficeCaltech(root=root_path,task="C",download=True,transform=trans)
        loader = DataLoader(ds, shuffle=True, batch_size=32)
        curr_sentences = SENTENCES[eval_ds]
        tokenized = processor.tokenizer(curr_sentences, padding=True, return_tensors='pt')
        tokenized['input_ids'], tokenized['attention_mask'] = tokenized['input_ids'].to(computation), \
                                                              tokenized['attention_mask'].to(computation)
        encoded = model.get_text_features(**tokenized)
        print("******** Testing the model ********")
        corrects, total = 0, 0
        for images, targets in loader:
            images, targets = images.to(computation), targets.to(computation)
            image_encoding = model.get_image_features(images)
            image_encoding /= image_encoding.norm(dim=-1, keepdim=True)
            encoded /= encoded.norm(dim=-1, keepdim=True)
            similarities = F.softmax((image_encoding @ encoded.T) / temperature)
            preds = torch.argmax(similarities, dim=1)
            corrects += torch.sum(preds == targets)
            total += len(targets)

    return corrects / total