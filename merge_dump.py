import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-time", type=float, help="last time step in old dump file")
parser.add_argument("-fil", type=str, help="old dump file name")
parser.add_argument("-append", type=str, help="dump file name to append")
args = parser.parse_args()
max_time = int(args.time / 500000) * 500000
# print max_time

time = 0
with open(args.fil) as f:
    for t, l in enumerate(f):
        if (t % 22118 == 0):
#             print time
            time = time + 1000
        if ( time > max_time ):
            break
        print l,

with open(args.append) as f:
    for t, l in enumerate(f):
        if ((t-1) % 22118 == 0):
            print "Atoms. Timestep: {:d}".format(time)
            time = time + 1000
        else:
            print l,



