#!/bin/bash --login

STRING=$(tail -n 22117 $1 | head -n 1)
echo $STRING | awk -v N=$3 '{print $3}'
