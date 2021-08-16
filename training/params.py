import argparse

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name == "ViT-B/32":
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorted", default=False, action="store_true", help="Sorts the training samples by length.")
    parser.add_argument("--train-data", type=str, default=None, help="Path to csv with training data paths.",)
    parser.add_argument("--cc_root", type=str, default=None, help="Path to the root of cc images.")
    parser.add_argument("--val-data", type=str, default=None, help="Path to csv file with validation data",)
    parser.add_argument("--imagenet-val", type=str, default=None, help="Path to imagenet val set.",)
    parser.add_argument("--logs", type=str, default="./logs/",
                        help="Where to store tensorboard logs. Use None to avoid storing logs.",)
    parser.add_argument("--name", type=str, default=None, help="Identifier for experiments. If not specified use current time.",)
    parser.add_argument("--workers", type=int, default=1, help="Number of workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of train epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=5000, help="Number of warmup steps.")
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU to run on (if None, uses all).",)
    parser.add_argument("--skip-scheduler", action="store_true", default=False, help="Skips weight decay.",)
    parser.add_argument("--save-frequency", type=int, default=1, help="How often to save checkpoints.")
    parser.add_argument("--zeroshot-frequency", type=int, default=2, help="How often to run zero shot.")
    parser.add_argument("--regression-frequency", type=int, default=2, help="How often to run zero shot.")
    parser.add_argument("--resume", default=None, type=str, help="Checkpoint to resume from.",)
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp", help="Precision.")
    parser.add_argument("--model", choices=["RN50x4", "RN50", "RN101", "ViT-B/32"], default="RN50",
                        help="Vision model to use.",)
    parser.add_argument("--skip-aggregate", default=False, action="store_true", help="Aggregates features before loss.")
    parser.add_argument("--tensorboard", default=True, action="store_true", help="Report results to tensorboard.")
    parser.add_argument("--wandb", default=False, action="store_true", help="Report results to wandb")
    parser.add_argument("--wandb-notes", default='', type=str,  help="Notes if logging with wandb")
    parser.add_argument("--C", type=float, default=3.16, help="inverse regularizer for logistic reg.")
    parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
    parser.add_argument("--multigpu", required=True, type=lambda x: [int(a) for a in x.split(",")],help="Which GPUs for multi-gpu",)
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.train_data is not None:
        assert args.cc_root is not None, "If you train on conceptual caption, need a root for the images."


    return args

if __name__ == "__main__":

    args = parse_args()
    print("Hey")