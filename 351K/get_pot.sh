#!/bin/bash --login

biaslist=$(seq -0.8500 0.05 -0.2000)

for bias in ${biaslist}
do
    python make_pot.py -fil log${bias} > pot${bias}
done
