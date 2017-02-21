#!/bin/bash --login

biaslist=$(seq $1 0.05 $2)

for bias in ${biaslist}
do
	python zcos_th.py -bias ${bias}
done
