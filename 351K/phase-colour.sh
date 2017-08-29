#!/bin/bash --login

# biaslist=$(seq $1 0.025 $2)
biaslist=( -0.8250 -0.3750 0.1000 )

# for b in ${biaslist}
for b in "${biaslist[@]}"
do
    python less_frame.py -bias ${b} -step 50 > sparse${b}.xyz
    ./strip_dump.sh sparse${b}.xyz stripped${b}
    python colour_phase.py -f stripped${b} -div 0.77777778 > anti-gauche${b}.xyz
#     python colour_phase.py -f stripped${b} -div 0.575 > O-DO${b}.xyz
    rm stripped${b} sparse${b}.xyz
done
