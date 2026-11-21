#!/bin/bash
# 2026-11-21: RFoldSearch v1.0.2 runtime ladders (top_n 500, as the paper) and
# peak memory (VmHWM) at full database size for RFoldSearch v1.0.2 and
# RiboSeek 0.2.1 RS-80. One job at a time, one core (taskset -c 3), threads 1.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=/root/autodl-tmp/rfoldseek; B=/root/autodl-tmp/bench2
PB=$R/envs/baselines/bin/python; PR=$R/envs/rfs/bin/python
LOG=$B/run_v102.log
hwm () {  # pid hwmfile: sample peak RSS until the process exits
    while kill -0 $1 2>/dev/null; do grep VmHWM /proc/$1/status > $2 2>/dev/null; sleep 2; done
}
rfs_args () {
    if [ $1 = rhofold ]; then echo "rhofold_db_83156.pt --subset-ids subsets --queries subsets/queries_1000.json"; else echo "nufold_db_531888_cbf966d.pt"; fi
}
cd $B
O=results_v102_top500; mkdir -p $O
for c in rhofold nufold; do
    echo "### ladder rfoldsearch v1.0.2 top_n=500 $c $(date +%T)" >> $LOG
    PYTHONPATH=$B/pkgs/rfs_v102 RFS_TOP_N=500 taskset -c 3 $PR bench_speed_v102.py $c $(rfs_args $c) $O --modes tm_a > $O/$c.log 2>&1 || echo "FAILED ladder $c" >> $LOG
    grep "n=" $O/$c.log | tail -1 >> $LOG
done
M=results_v102_mem; mkdir -p $M
for c in rhofold nufold; do
    if [ $c = rhofold ]; then N=83156; else N=531888; fi
    echo "### memory rfoldsearch v1.0.2 top_n=500 $c full $(date +%T)" >> $LOG
    PYTHONPATH=$B/pkgs/rfs_v102 RFS_TOP_N=500 taskset -c 3 $PR bench_speed_v102.py $c $(rfs_args $c) $M/$c --sizes $N --modes tm_a > $M/rfs_$c.log 2>&1 &
    P=$!; hwm $P $M/rfs_$c.hwm; wait $P; echo "EXIT $?" >> $M/rfs_$c.log
    echo "rfs $c $(cat $M/rfs_$c.hwm)" >> $LOG
done
cd $R
for c in rho nu; do
    echo "### memory riboseek 0.2.1 rs80 $c full $(date +%T)" >> $LOG
    NU_SIZES=531888 RHO_SIZES=83156 taskset -c 3 $PB rs021_ref_sizes.py --time $c --alphabet rs80 > $B/$M/rs80_$c.log 2>&1 &
    P=$!; hwm $P $B/$M/rs80_$c.hwm; wait $P; echo "EXIT $?" >> $B/$M/rs80_$c.log
    echo "rs80 $c $(cat $B/$M/rs80_$c.hwm)" >> $LOG
done
echo ALL_DONE $(date +%T) >> $LOG
