#!/bin/bash --login

STRING=$(tail -n 22117 $1 | head -n 1)
TIME=$(echo $STRING | awk -v N=$3 '{print $3}')

python merge_dump.py -time $TIME -fil $1 -append $2 > dump-soft-0.7250.xyz
