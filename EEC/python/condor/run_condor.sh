#!/bin/bash

cd /afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python/

echo "Arguments passed to this script are: "
echo "  for 1 (script): $1"
echo "  for 2 (input filename): $2"
echo "  for 3 (output filename): $3"

python3 ${1} ${2} ${3}

#mv ${3} /eos/user/z/zhangj/ALEPH/EECNtuples/
