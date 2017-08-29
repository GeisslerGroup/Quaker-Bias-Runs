from __future__ import division
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-phase", type=str, help="phase value for which to get lesser time frames")
parser.add_argument("-step", type=int, help="every step-th time frame to be used")
args = parser.parse_args()

n_atoms = 22116
n_lines = n_atoms + 2

# with open("dump-nospring-" + args.phase + ".xyz") as f:
with open("dump-moving" + args.phase + ".xyz") as f:
    for n, l in enumerate(f):
        if (n//n_lines) % args.step == 0:
            print l,

