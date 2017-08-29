#!/bin/bash --login

biaslist=( -0.7750 -0.6750 -0.5750 -0.5250 -0.4750 -0.4250 -0.1500 -0.1000 -0.0500 0.0000 )
biaslist=( -0.7750 -0.6750 -0.5750 -0.5250 -0.4750 -0.4250 -0.3750 )
biaslist=( -0.8250 -0.3750 0.1000 )

for bias in "${biaslist[@]}"
do
    STRING=$(tail -n 22117 dump-soft${bias}.xyz | head -n 1)
    TIME=$(echo $STRING | awk -v N=$3 '{print $3}')
    python merge_dump.py -time $TIME -fil dump-soft${bias}.xyz -append dump-soft-more${bias}.xyz > dump-full${bias}.xyz
done

for bias in "${biaslist[@]}"
do
    STRING=$(tail -n 22117 dump-soft${bias}.xyz | head -n 1)
    TIME=$(echo $STRING | awk -v N=$3 '{print $3}')
    python merge_log.py -time $TIME -fil log${bias} -append log-more${bias} > pot${bias}
done

