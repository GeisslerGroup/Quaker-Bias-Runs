#!/bin/bash --login

convert -crop +345+260 disord-O-DO-surf.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-disord-O-DO-surf.tga

convert -crop +345+260 disord-O-DO-vdw.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-disord-O-DO-vdw.tga

convert -crop +345+260 ord-O-DO-surf.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-ord-O-DO-surf.tga

convert -crop +345+260 ord-O-DO-vdw.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-ord-O-DO-vdw.tga

convert -crop +345+260 disord-a-g-surf.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-disord-a-g-surf.tga

convert -crop +345+260 disord-a-g-vdw.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-disord-a-g-vdw.tga

convert -crop +345+260 ord-a-g-surf.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-ord-a-g-surf.tga

convert -crop +345+260 ord-a-g-vdw.tga test.tga
convert -crop -345-260 test.tga test1.tga
mv test1.tga cropped-ord-a-g-vdw.tga

rm test.tga



