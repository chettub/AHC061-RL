#!/bin/bash

SEEDS_FILE="tools/seeds.txt"
if [ -f "$SEEDS_FILE" ]; then
    rm "$SEEDS_FILE"
fi
for i in {0..30000}
do
    echo $i >> "$SEEDS_FILE"
done

# build tools and generate input files
cd tools
cargo build --release
./target/release/gen seeds.txt
cd ..
