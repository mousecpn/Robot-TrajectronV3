#!/bin/bash

# pkill -f simulated_shared_benchmark.py "lowdof"
# looping through different user types and methods  "noisy" "laggy"  "modeswitching" "singledof" "ho"  "laggy" 
USER_TYPES=("noisy" "laggy" "modeswitching" "singledof") 
METHODS=("rt")
OOD_ALPHA=(0.85)
HISTORY_SIZE=(14)
SIGMA_COFF=(1)
DEVICE=0
DEVICES=$(( $(nvidia-smi --list-gpus | wc -l) ))
for USER in "${USER_TYPES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for ALPHA in "${OOD_ALPHA[@]}"; do
            for H_SIZE in "${HISTORY_SIZE[@]}"; do
                for SIGMA in "${SIGMA_COFF[@]}"; do
                    echo "Running benchmark with user: $USER, method: $METHOD, ood_alpha: $ALPHA, history_size: $H_SIZE, and sigma_coff: $SIGMA"
                    CUDA_VISIBLE_DEVICES=$DEVICE python simulated_shared_benchmark.py --user "$USER" --method "$METHOD" --ood_alpha "$ALPHA" --history_size "$H_SIZE" --sigma_coff "$SIGMA" --no-debug &
                    DEVICE=$(( (DEVICE + 1) % DEVICES ))
                done
                # echo "Running benchmark with user: $USER, method: $METHOD, ood_alpha: $ALPHA, and history_size: $H_SIZE"
                # CUDA_VISIBLE_DEVICES=$DEVICE python simulated_shared_benchmark.py --user "$USER" --method "$METHOD" --ood_alpha "$ALPHA" --history_size "$H_SIZE" --no-debug &
                # DEVICE=$((1-$DEVICE))
            done
        done
    done
done
wait
echo "All simulated shared benchmark jobs finished."