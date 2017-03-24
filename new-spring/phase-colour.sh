#!/bin/bash --login

biaslist=$(seq -0.9500 0.0250 -0.7000)

for bias in ${biaslist}
do
	./strip_dump.sh dump-long${bias}.xyz long.stripped.${bias}
	python colour_phase.py -f long.stripped.${bias} -div 0.77777778 > long${bias}.xyz
done
