from crystal_cifar10 import crystal_cifar10
import os
import datetime
import argparse
from warnings import simplefilter

if __name__ == '__main__':
    config = config()

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('type', type=str, default="DB+DC+GA")
    # parser.add_argument('cuda', type=int, default=0)
    # parser.add_argument('dataset', type=str, default="CIFAR10")
    # parser.add_argument('white', type=int, default=0) # if white-box
    # parser.add_argument('fed', type=int, default=1) # if crystal-box
    # args = parser.parse_args()

    # types = args.type
    # cuda = args.cuda
    # config.set_subkey("general", "cuda", cuda)
    # config.set_subkey("general", "type", types)
    # config.set_subkey("general", "white", args.white)
    # config.set_subkey("general", "fed", args.fed)
    # config.set_subkey("general", "type", types)


    print("CIFAR10")
    config.set_subkey("statistics", "dataset", "CIFAR10")
    config.set_subkey("learning", "epochs", 40)
    config.set_subkey("learning", "learning_rate", 0.001)
    config.set_subkey("general", "number_shadow_model", 10)
    config.set_subkey("general", "test_target_size", 5000)
    config.set_subkey("general", "train_target_size", 5000)
    config.set_subkey("general", "classes", 10)

    crystal_cifar10(config)



