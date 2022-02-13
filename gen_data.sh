#!/bin/bash

cd ../build
make

rm -r ../output
cd ..
mkdir output
cd build

counter=1
ITERATIONS=100

while [ $counter -le $ITERATIONS ]
do
    echo
    echo "["$counter"/"$ITERATIONS"]"
    echo
    time ./rrt_star "${counter}"
    ((counter++))
done

echo All done