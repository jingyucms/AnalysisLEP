#!/bin/bash

cd /afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/unfolding/

source RooUnfold/build/setup.sh

echo "Arguments passed to this script are: "
echo "  for 1 (script): $1"
echo "  for 2 (input filename): $2"
echo "  for 3 (input filename): $3"
echo "  for 4 (input filename): $4"

python3 ${1} ${2} ${3} ${4} ${5}
