#!/bin/bash


N=64
k='20'
m=64
maxq=0.1
q_method='G'
noise_level=0.0
# 以上是一些预设，可设置为benchmark

ARGNAME='m'  
# 当考虑ARGNAME这一变量作变化时 
# ARGNAME:k/m/maxq/noise_level


Opt () {
    m=$1 &&    # 更改ARGNAME时也要修改这里
    TITLE=$ARGNAME$1'_'$q_method'_'$2 &&
    echo > ${TMP_PATH} &&
    nohup python -u \
    ${MAIN_PATH} \
    --PROJECT_DIR ${PROJECT_DIR} \
    --N ${N} \
    --k ${k} \
    --m ${m} \
    --maxq ${maxq} \
    --q_method ${q_method} \
    --noise_level ${noise_level} \
    --forward_solver $2 \
    --title ${TITLE} \
    --output_filename ${OUTPUT_LOG} \
    --Net_name 'k'${k}'_P_4,64,uniform_G_0.1_NST_R200_12,32,4_1' \
    >> ${TMP_PATH} 2>&1 &&
    python -u ${WRITE_PATH} \
    --output_filename ${OUTPUT_LOG}
}


PROJECT_DIR='/data/liuziyang/Programs/pde_solver/'
PYDIR=${PROJECT_DIR}'src/'
LOGDIR=${PROJECT_DIR}'logs/'
OUTPUT_LOG=$LOGDIR'output_'$ARGNAME'_'$q_method'.log'
MAIN_PATH=${PYDIR}'Main.py'
WRITE_PATH=${PYDIR}'Write_J.py'
DRAW_PATH=${PYDIR}'draw.py'
TMP_PATH='.tmp.log'
RES_DIR=${DIR}'pic/res/'

echo > ${OUTPUT_LOG} &&
Opt '16' 'MUMPS' &&
Opt '16' 'NET' &&
Opt '32' 'MUMPS' &&
Opt '32' 'NET' &&
Opt '64' 'MUMPS' &&
Opt '64' 'NET' &&


python ${DRAW_PATH} \
--logname ${OUTPUT_LOG} \
--argname ${ARGNAME} \
--savepath ${RES_DIR} \
--q_method ${q_method} &


# tail -f .tmp.log 来查看运算进度