from __future__ import division
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bias", type=str, help="bias value for which to get lesser time frames")
parser.add_argument("-step", type=int, help="every step-th time frame to be used")
args = parser.parse_args()

n_atoms = 22116
n_lines = n_atoms + 2

with open("dump-moving" + args.bias + ".xyz") as f:
    for n, l in enumerate(f):
        if (n//n_lines) % args.step == 0:
            print l,

