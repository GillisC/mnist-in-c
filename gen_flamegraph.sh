#!/bin/bash

make clean
make

sudo perf record -F 99 -g -- ./build/main
sudo perf script | ./FlameGraph/stackcollapse-perf.pl > out.folded
./FlameGraph/flamegraph.pl out.folded > flamegraph.svg


