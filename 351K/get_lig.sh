#!/bin/bash --login

biaslist=$(seq $1 $2 $3)

for bias in ${biaslist}
do
	awk '(/^2/ || /^1/) && $3 >= 0' dump-full${bias}.xyz > lig.${bias}
done
