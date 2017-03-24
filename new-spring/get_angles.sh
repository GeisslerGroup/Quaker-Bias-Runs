#!/bin/bash --login

biaslist=$(seq $1 $2 $3)

for bias in ${biaslist}
do
	python zcos_th.py -bias ${bias} -start 0
done
