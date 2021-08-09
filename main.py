import argparse
from train import train_model
import matplotlib.pyplot as plt

def main(**kwargs):

    model, accuracies = train_model(**kwargs)
    plt.figure()
    plt.plot(accuracies)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Curriculum CLIP")
    parser.add_argument('-i', '--images_path', type=str, help="Path to where the images of the dataset are kept",
                        required = True)
    parser.add_argument('-a', '--annotation_path', type=str, help="Path to where the annotations of the dataset are kept",
                        required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Size of the batch you want to train on.")
    parser.add_argument('-e', '--epoch_nums', type=int, default=100, help="Num of training epochs.")
    parser.add_argument('-r', '--run_on', type=str, default='cuda', help="Whether you want to run on GPU or CPU")
    parser.add_argument('-l', '--lr', type=float, default=0.01, help="Learning of the optimizer")
    parser.add_argument('-m', '--momentum', type=float, default=0.8, help="Momentum of the optimizer")
    parser.add_argument('-s', '--strategy', type=str, default="max", help="Strategy of the curriculum")
    parser.add_argument('-t', '--temperature', type=float, help="Temperature of the CLIP model.")
    parser.add_argument('--root_path', type=str, help="Where to keep the OfficeCaltech data.")
    parser.add_argument('--resnet_size', type=int, help="Size of the resnet visual encoder.")
    parser.add_argument('--eval_ds', type=str, default='cifar')
    args = parser.parse_args()
    main(**vars(args))