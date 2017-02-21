#!/bin/bash --login

biaslist=$(seq $1 0.05 $2)

for b in ${biaslist}
do
    python less_frame.py -bias ${b} -step 10 > sparse${b}.xyz
    ./strip_dump.sh sparse${b}.xyz sparse${b}.stripped
    python colour_phase.py -f sparse${b}.stripped -div 0.77777778 > colour${b}.xyz
    mv sparse${b}.xyz sparse${b}.stripped colour${b}.xyz sparse-files/
done
