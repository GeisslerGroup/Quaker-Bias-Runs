#!/bin/bash --login

biaslist=( -0.6750 -0.5250 -0.4750 -0.4250 -0.3750 )

for bias in "${biaslist[@]}"
do
    STRING=$(tail -n 22117 dump-moving${bias}.xyz | head -n 1)
    TIME=$(echo $STRING | awk -v N=$3 '{print $3}')
    NEWT=$(echo $(((($TIME/1000 - 1000)*22118) / 22118)) | bc)
    EQTIME=$(echo $((($NEWT-5000) / 10)) | bc)
    NLINES=$(echo $(($EQTIME*22118 + ($NEWT-$EQTIME*10)*22118)) | bc)
    head -n $NLINES dump-moving${bias}.xyz > dump-smaller${bias}.xyz
done

for bias in "${biaslist[@]}"
do
    mv dump-smaller${bias}.xyz dump-moving${bias}.xyz
done
