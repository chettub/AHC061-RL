#!/bin/bash
n=$1
formatted_n=$(printf "%04d" $n)
./tools/target/release/tester ./main < "tools/in/${formatted_n}.txt" > out
