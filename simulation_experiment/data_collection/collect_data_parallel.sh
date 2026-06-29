#!/bin/bash

# 并行进程数
NUM_JOBS=10

# 每个进程处理的 scene 数
SCENES_PER_JOB=2000  # 根据你数据集大小修改
START_SCENE=0
# Python 脚本文件名（替换成你的真实文件名）
PYTHON_SCRIPT="data_collection/main.py"

# 公共参数
SAVE_DIR="./data/trajectory/trajectories_pregrasp.npz"
SAVE_INTERVAL=1000
NO_DEBUG="--no-debug"  # 如果你想打开GUI，就设成空字符串

pkill -f main.py

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 启动并行进程
for ((i=0; i<$NUM_JOBS; i++))
do
    CURRENT_SCENE=$((START_SCENE + i * SCENES_PER_JOB))
    echo "Launching job $i: scenes $CURRENT_SCENE to $((CURRENT_SCENE + SCENES_PER_JOB - 1))"

    JOB_SAVE_NAME="./data/trajectory/trajectories_pregrasp_job_${i}.npz"

    python $PYTHON_SCRIPT \
        --save_file_name "$JOB_SAVE_NAME" \
        --save_interval $SAVE_INTERVAL \
        --start-scene $CURRENT_SCENE \
        --num-scenes $SCENES_PER_JOB \
        $NO_DEBUG \
        &  
        # > "job_$i.log" 2>&1 &  # 日志输出到文件
done

# 等待所有任务完成
wait

echo "All parallel jobs finished."
