#!/bin/bash
set -x

log_dir=$HOME/logs
script=$1
job_id=$(sbatch $script | grep -Eo '[0-9]{1,}')
log_err=$log_dir/$job_id.err
log_out=$log_dir/$job_id.out
sleep 2s
while true; do
    if [ -f "$log_err" ]; then
        tail -f $log_err -f $log_out
    else 
        echo "==> $log_err and $log_out do not exist yet, wait for 5s..."
        sleep 5s
    fi
done
