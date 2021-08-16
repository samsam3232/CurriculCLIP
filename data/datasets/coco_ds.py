import torchvision.datasets as ds
from typing import Any, Callable, Optional, Tuple, List
import json
from tqdm import tqdm


class CurriculumCocoCaptions(ds.CocoCaptions):

    def __init__(self, root: str, annFile: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None, strategy: str = "max",
                 **kwargs):
        super().__init__(root=root, annFile=annFile, transforms=transforms,
                         transform=transform, target_transform=target_transform)

        self.strategy = strategy
        self.ids = self.retrieve_ids(annFile=annFile, strategy = strategy)

    def retrieve_ids(self, annFile: str, strategy: str = "max"):

        def get_unique_dims():

            unique_dims = dict()
            for i in tqdm(data['images']):
                if (i['height'], i['width']) in unique_dims:
                    unique_dims[(i['height'], i['width'])] += 1
                else:
                    unique_dims[(i['height'], i['width'])] = 1

            sorted_dims = sorted(unique_dims, key=unique_dims.get)  # [1, 3, 2]
            filtered_sorted_dims = [i for i in sorted_dims[-20:] if i[0] < i[1]]

            return filtered_sorted_dims

        def get_images_dict():

            images_dict = dict()
            for i in data['images']:
                images_dict[i['id']] = i

            return images_dict

        def check_criterion(dic, sample):

            if curr_strat == "min":
                return len(sample['caption'].split(' ')) < len(dic[sample['image_id']].split(' '))
            elif curr_strat == "max":
                return len(sample['caption'].split(' ')) < len(dic[sample['image_id']].split(' '))

        def get_annotations_dict():

            annotations_dict = dict()
            for i in tqdm(data['annotations']):
                if (i['image_id'] in annotations_dict) and (check_criterion(annotations_dict, i)):
                    annotations_dict[i['image_id']] = i['caption']
                elif not (i['image_id'] in annotations_dict) and ((img_dict[i['image_id']]['height'],
                                                                       img_dict[i['image_id']][
                                                                           'width']) in sorted_unique_dims):
                    annotations_dict[i['image_id']] = i['caption']

            lengths = dict()
            for i in annotations_dict.keys():
                lengths[i] = (len(annotations_dict[i].split(" ")))

            sorted_lengths = sorted(lengths, key=lengths.get)

            return sorted_lengths

        with open(annFile, 'r') as f:
            data = json.load(f)

        curr_strat = strategy
        sorted_unique_dims = get_unique_dims()
        img_dict = get_images_dict()
        lengths = get_annotations_dict()

        return lengths

    def _load_target(self, id) -> str:
        return super()._load_target(id)[0]