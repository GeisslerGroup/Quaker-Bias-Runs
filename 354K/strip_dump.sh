#!/bin/bash --login

awk 'NR != 1' $1 > dump1.xyz
awk 'NR % 22118 != 0' dump1.xyz > dump2.xyz
awk 'NR != 1' dump2.xyz > dump3.xyz
awk 'NR % 22117 != 0' dump3.xyz > dump4.xyz
rm dump1.xyz dump2.xyz dump3.xyz
mv dump4.xyz $2 

