#!/bin/bash --login

biaslist=$(seq -0.8500 0.025 0.1000)

for bias in ${biaslist}
do
    mv pot${bias} pot-351.${bias}
done
