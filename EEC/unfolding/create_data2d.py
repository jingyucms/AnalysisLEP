import ROOT
import numpy as np
import sys
import os
import argparse

eijbins = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

sys.path.append(os.path.abspath("/afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python"))

from analysis_eec import *

#xlow = -3
#xhigh = np.log10(np.pi/2)
#nbins = 100

#width = (xhigh-xlow)/nbins

#bins=[]
#for i in range(nbins+1):
#    val = pow(10, xlow + i * width)
#    bins += [val]

#newbins = [np.pi- b for b in bins]
#newbins = newbins[::-1]
#del newbins[0]

#bin_edge = bins+newbins

#bin_edge = calcBinEdge(0.002, np.pi/2, 100)
bin_edge = calcBinEdge(0.00001, 0.5, 100)

datafile = '/eos/user/z/zhangj/ALEPH/EECNtuples/h_LEP1Data1994_recons_aftercut-MERGED.root'

outfile = 'data_LEP1MC1994_z_v6.root'

fin = ROOT.TFile.Open(datafile,'r')
fout = ROOT.TFile.Open(outfile,'recreate')

reco2d = ROOT.TH2D("EEC_2d", "", 200, np.array(bin_edge), len(eijbins)-1, np.array(eijbins))
tmc = fin.Get('eec')
tmc.Project('EEC_2d', 'eec:z')

counter = fin.Get('N')

print(reco2d.Integral())

fout.cd()
reco2d.Write()
counter.Write()
