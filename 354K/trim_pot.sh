#!/bin/bash --login

biaslist=$(seq -0.8500 0.025 0.1000)

for bias in ${biaslist}
do
    tail -n 5000 pot${bias} > pot-354${bias}
done
