import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

def draw_curve(fname):
    f = pickle.load(open(fname, 'rb'))
    x = np.linspace(1000,60000,60)
    y = np.array(f)
    plt.plot(x,y)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    rl_type = fname.split('/')[2].split('.')[0].split('1')[0]
    plt.title(rl_type)
    plt.savefig(rl_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type = str)
    args = parser.parse_args()
    fname = osp.join("../learning_curves", args.name)
    draw_curve(fname)
