#!/bin/bash --login

python less_frame.py -bias 0.65 -step 500 > sparse0.65.xyz
./strip_dump.sh sparse0.65.xyz sparse0.65.stripped
python colour_phase.py -f sparse0.65.stripped -div 0.62 > colour0.65.xyz
