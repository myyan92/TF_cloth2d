from train_eval import Trainner
from eval_visualize import Visualizer
import gin, argparse
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=["train", "eval", "visualize"],
                        help="running train or eval.")
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)
    if args.action == "train":
        trainner = Trainner()
        trainner.train()
    elif args.action == "eval":
        trainner = Trainner()
#        trainner.model.load(trainner.sess, trainner.snapshot)
        trainner.eval()
    else:
        vis = Visualizer()
        vis.eval()

