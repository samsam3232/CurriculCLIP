import argparse
import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image

ROOT = "cc_data"

def grab(row_data):
    """
    Downloads images from the TSV file.
    """

    row_id, line, split, root = row_data

    try:
        caption, url = line.split("\t")[:2]
    except:
        print("Parsing error")
        return None

    if os.path.exists(os.path.join(root, f"{split}/{row_id % 1000}/{row_id}.jpg")):
        print("Finished: ", row_id)
        return row_id, caption, url

    try:
        res = requests.get(url, timeout=40)
        if res.status_code != 200:
            print("Some error occured while requesting the url. Probably 404", url)
            return None

        im = Image.open(BytesIO(res.content))
        im.thumbnail((512, 512), PIL.Image.BICUBIC)
        if min(*im.size) < max(*im.size) / 3:
            print("Ratio is too uneven:  ", url)
            return None
        im.save(os.path.join(root, f"{split}/{row_id % 1000}/{row_id}.jpg"))

        try:
            o = Image.open(os.path.join(root, f"{split}/{row_id % 1000}/{row_id}.jpg"))
            o = np.array(o)

            print("Success", o.shape, row_id, url)
            return row_id, caption, url
        except:
            print("Failed", row_id, url)

    except Exception as e:
        print("Unknown error", e)
        pass


def create_dirs(root):

    if not os.path.exists(root):
        os.mkdir(root)
        os.mkdir(root + "/train")
        os.mkdir(root + "/val")
        for i in range(1000):
            os.mkdir(root + "/train/" + str(i))
            os.mkdir(root + "/val/" + str(i))


def main(root, input_tsv_paths, output_path):

    if len(output_path) == 0:
        has_outs_dir = False
    else:
        has_outs_dir = True
        assert len(input_tsv_paths) == len(output_path), "If you define one output path, then you need to define one for," \
                                                        "every input tsv"

    create_dirs(root)
    processes = mp.Pool(300)

    for i in range(len(input_tsv_paths)):

        tsv_path = input_tsv_paths[i]

        if not ('val' in tsv_path.lower() or 'train' in tsv_path.lower()):
            print(f"Can't know if {tsv_path} val or train set, skipping it")
            continue
        print(f"Currently processing the following file {tsv_path} \n \n")

        split = 'val' if 'val' in tsv_path.lower() else 'train'
        results = processes.map(grab, [(i, split, x, root) for i, x in enumerate(open(tsv_path).read().split("\n")[:5000])])

        if has_outs_dir:
            out = open(os.path.join(output_path[i], tsv_path.split("/")[-1]), "w")
        else:
            out = open(tsv_path.replace(".tsv", "_output.csv"), "w")
        out.write("title\tfilepath\n")

        for row in results:
            print("Processing row: ", row)
            if row is None:
                continue
            row_id, caption, url = row
            file_path = os.path.join(root, f"{split}/{str(row_id % 1000)}/{row_id}.jpg")
            if os.path.exists(file_path):
                out.write(f"{caption}\t{file_path}\n")
            else:
                print("Dropped id: ", id)
        out.close()

    processes.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data processing and dowloading")
    parser.add_argument("-r", "--root", type=str, default=ROOT, help="Path to the root where the data is to be kept")
    parser.add_argument("-i", "--input_tsv_paths", type=str, nargs='+', required=True)
    parser.add_argument("-o", "--output_path", type=str, nargs='*', default=[])
    args = parser.parse_args()
    main(**vars(args))