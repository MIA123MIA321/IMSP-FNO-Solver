#!/bin/bash


N=64
N_comp='64'
k='20'
m=32
maxq=0.1
q_method='TEST'
noise_level=0.0
# 以上是一些预设，可设置为benchmark

ARGNAME='k'  
# 当考虑ARGNAME这一变量作变化时 
# ARGNAME:k/m/maxq/noise_level


Opt () {
    k=$1 &&    # 更改ARGNAME时也要修改这里
    TITLE=$ARGNAME$1'_'$q_method'_'$2 &&
    echo > ${TMP_PATH} &&
    nohup python -u \
    ${MAIN_PATH} \
    --PROJECT_DIR ${PROJECT_DIR} \
    --N $3 \
    --N_comp $4 \
    --k ${k} \
    --m ${m} \
    --maxq ${maxq} \
    --q_method ${q_method} \
    --noise_level ${noise_level} \
    --forward_solver $2 \
    --title ${TITLE} \
    --output_filename ${OUTPUT_LOG} \
    >> ${TMP_PATH} 2>&1 &&
    python -u ${WRITE_PATH} \
    --PROJECT_DIR ${PROJECT_DIR} \
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
# Opt '40' 'MUMPS' 64 '64' &&
# Opt '40' 'MUMPS' 64 '64' &&
Opt '20,40' 'MUMPS,MUMPS' 128 '64,128' &&
Opt '20,40' 'NET,MUMPS' 128 '64,128' &&
# Opt '40' 'MUMPS' 128 '128' &&
# Opt '20,80' 'NET,MUMPS' 128 '64,128' &&
# Opt '80' 'MUMPS' 128 '128' &&

python ${DRAW_PATH} \
--logname ${OUTPUT_LOG} \
--argname ${ARGNAME} \
--savepath ${RES_DIR} \
--q_method ${q_method} &


# tail -f .tmp.log 来查看运算进度