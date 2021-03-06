import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

def draw_curve(fname):
    f = pickle.load(open(fname, 'rb'))
    # x = np.linspace(200, 6000, 30)
    y = np.array(f)
    # y = y[1::3]
    x = np.linspace(50, 50 * len(y), len(y))
    plt.plot(x,y)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    rl_type = fname.split('/')[2].split('.')[0].split('1')[0]
    plt.title(rl_type)
    plt.savefig(rl_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='full_tag_rewards.pkl')
    args = parser.parse_args()
    fname = osp.join("../learning_curves", args.name)
    draw_curve(fname)
