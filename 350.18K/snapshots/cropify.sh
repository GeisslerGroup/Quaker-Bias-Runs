#!/bin/bash --login

biaslist=( -0.7500 -0.7250 -0.6500 -0.5750 -0.5000 -0.4250 -0.3500 -0.2750 -0.2000 -0.1750 )

for bias in "${biaslist[@]}"
do
#     echo ${bias}
    convert -crop +345+270 O-DO${bias}-surf.tga test.tga
    convert -crop -345-270 test.tga test1.tga
    mv test1.tga cropped-O-DO${bias}-surf.tga
    convert -crop +345+270 O-DO${bias}-vdw.tga test.tga
    convert -crop -345-270 test.tga test1.tga
    mv test1.tga cropped-O-DO${bias}-vdw.tga
    rm test.tga
done

newlist=( -0.7500 -0.7250 -0.6500 -0.5750 -0.5000 -0.4250 -0.2750 -0.2000 )

for bias in "${newlist[@]}"
do
#     echo ${bias}
    convert -crop +345+270 a-g${bias}-surf.tga test.tga
    convert -crop -345-270 test.tga test1.tga
    mv test1.tga cropped-a-g${bias}-surf.tga
    convert -crop +345+270 a-g${bias}-vdw.tga test.tga
    convert -crop -345-270 test.tga test1.tga
    mv test1.tga cropped-a-g${bias}-vdw.tga
    rm test.tga
done




