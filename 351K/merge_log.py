import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-time", type=float, help="last time step in old log file")
parser.add_argument("-fil", type=str, help="old log file name")
parser.add_argument("-append", type=str, help="log file name to append")
args = parser.parse_args()
max_time = int(args.time / 500000) * 500000
# print max_time

time = 0
with open(args.fil) as f:
    for t, l in enumerate(f):
        time = time + 1000
        if ( time > max_time ):
            break
        l_arr = l.split()
        print "{} {}".format(time, l_arr[5])

with open(args.append) as f:
    for t, l in enumerate(f):
        l_arr = l.split()
        print "{} {}".format(time, l_arr[5])
        time = time + 1000


