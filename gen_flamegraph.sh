#!/bin/bash

set -e

make clean
make

mkdir -p out

sudo perf record -F 99 -g -o out/perf.data  -- ./build/main
sudo perf script -i out/perf.data | ./FlameGraph/stackcollapse-perf.pl > out/out.folded
./FlameGraph/flamegraph.pl out/out.folded > out/flamegraph.svg

echo "Flamegraph successfully generated at out/flamegraph.svg"


