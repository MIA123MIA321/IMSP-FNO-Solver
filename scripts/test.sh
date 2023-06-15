#!/bin/bash


N_rec=64
N_comp='64'
k='40'
N_src=32
maxq=0.1
q_method='TEST'
noise_level=0.0
bd_num=1
scheme='PML'
bd_type='N'
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
    --N_comp $3 \
    --k ${k} \
    --N_src ${N_src} \
    --maxq ${maxq} \
    --maxiter $4 \
    --q_method ${q_method} \
    --load_boundary $5 \
    --NS_length $6 \
    --noise_level ${noise_level} \
    --bd_type ${bd_type} \
    --forward_solver $2 \
    --title ${TITLE} \
    --bd_num ${bd_num} \
    --scheme ${scheme} \
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

Opt '20' 'MUMPS' '128' '15' 'F' 3 &&
Opt '20' 'NET' '64' '15' 'T' 3 &&


python ${DRAW_PATH} \
--logname ${OUTPUT_LOG} \
--argname ${ARGNAME} \
--savepath ${RES_DIR} \
--q_method ${q_method} &


# tail -f .tmp.log 来查看运算进度