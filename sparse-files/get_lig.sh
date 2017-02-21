#!/bin/bash --login

biaslist=$(seq -0.90 0.05 -0.65)

for bias in ${biaslist}
do
	awk '(/^2/ || /^1/) && $3 >= 0' sparse${bias}.stripped > lig.${bias}
done
