import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fil", type=str, help="log file name")
args = parser.parse_args()

with open(args.fil) as f:
    for t, l in enumerate(f):
        l_arr = l.split()
#         print "{} {}".format(l_arr[0], l_arr[5])
        print "{}".format(l_arr[5])


