#!/bin/bash
# Run SPONGE gradient tests and compare 1e gradient with PySCF
SPONGE=/Users/xiayj/Desktop/research/SPONGE/build-dev-cpu/SPONGE
BASE=/Users/xiayj/Desktop/research/SPONGE/benchmarks/comparison/tests/pyscf/statics

for case in h2_sto3g h2o_sto3g h2o_631g ch4_sto3g ch4_631g; do
    dir="$BASE/$case/sponge"
    if [ ! -d "$dir" ]; then
        echo "SKIP: $case (dir not found)"
        continue
    fi
    echo "=== $case ==="
    cd "$dir"
    $SPONGE -mdin mdin.txt 2>&1 | grep "QC ="
    cd - > /dev/null
done
