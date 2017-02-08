#!/bin/bash --login

biaslist=$(seq 0.35 0.05 0.95)

for b in ${biaslist}
do
    python less_frame.py -bias ${b} -step 500 > sparse${b}.xyz
    ./strip_dump.sh sparse${b}.xyz sparse${b}.stripped
    python colour_phase.py -f sparse${b}.stripped -div 0.8 > colour${b}.xyz
    mv sparse${b}.xyz sparse${b}.stripped colour${b}.xyz sparse-files/
done
