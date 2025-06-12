#!/bin/bash

export generator=genBefore

#for FILE_PATH in `ls /eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/*1994*.root`
#for FILE_PATH in `ls /eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPH/*1994*.root`
for FILE_PATH in `ls /eos/user/z/zhangj/ALEPH/SamplesLEP1/HERWIG7/run_00/*.root`
#for FILE_PATH in `ls /eos/user/z/zhangj/ALEPH/SamplesLEP1/SHERPA/run_00/*.root`
do
    FILE_NAME=$(basename "$FILE_PATH")
    echo ${FILE_PATH}
    echo ${FILE_NAME}
    export f=${FILE_NAME}
    export p=${FILE_PATH}
    condor_submit condor.sub
done
