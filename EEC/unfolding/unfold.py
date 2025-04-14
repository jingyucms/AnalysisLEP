import ROOT
import numpy as np
import sys
import os
import argparse

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

from create_response_matrices import *

def Proj2D_Y(h,xmin,xmax,hname="XXX"):

    # project 2D histogram into 1D along Y

    imin=h.GetXaxis().FindBin(xmin)
    imax=h.GetXaxis().FindBin(xmax)-1
    
    proj_y=h.ProjectionY(hname, imin, imax)
    ROOT.SetOwnership(proj_y,True)

    return proj_y

def Proj2D_X(h,ymin,ymax,hname="XXX",Debug=False):

    # project 2D histogram into 1D along Y

    imin=h.GetYaxis().FindBin(ymin)
    imax=h.GetYaxis().FindBin(ymax)-1

    proj_x=h.ProjectionX(hname, imin, imax)
    ROOT.SetOwnership(proj_x,True)

    return proj_x

eijbins = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

xlow = -3
xhigh = np.log10(np.pi/2)
nbins = 100

width = (xhigh-xlow)/nbins

bins=[]
for i in range(nbins+1):
    val = pow(10, xlow + i * width)
    bins += [val]

newbins = [np.pi- b for b in bins]
newbins = newbins[::-1]
del newbins[0]

bin_edge = bins+newbins

if __name__ == '__main__':

    #filenamein = "response_LEP1MC1994_v13.root"
    #filenamein = "response_LEP1MC1994_z_v11.root"
    #filenamein = "smeared_response_Herwig715.root"
    #filenamein = "smeared_response_Sherpa.root"
    filenamein = "smeared_response_LEP1MC1994.root"
    
    #datafile = 'data_LEP1MC1994_v6.root'
    #datafile = 'data_LEP1MC1994_z_v6.root'
    datafile = 'smeared_response_LEP1MC1994.root'
    
    #filenameout = "unfolded_data_v13_bin2.root"
    #filenameout = "unfolded_data_z_v11_bin2.root"
    #filenameout = "unfolded_LEP1MC1994_v11_bin2.root"
    #filenameout = "unfolded_smeared_Herwig715_bin2.root"
    #filenameout = "unfolded_smeared_Sherpa_bin2.root"
    filenameout = "unfolded_smeared_LEP1MC1994_bin2.root"

    fin = ROOT.TFile.Open(filenamein,'r')
    fdata = ROOT.TFile.Open(datafile,'r')
    
    response = fin.Get("response2d_eij_r_bin2")

    reco2d = fdata.Get('smr2d_eij_r_bin2')
    #reco2d = fdata.Get('EEC_2d')
    gen2d = fin.Get("gen2d_eij_r_bin2")

    normalization = fin.Get("counter").GetBinContent(2)
    n = fdata.Get('counter').GetBinContent(2)

    print(normalization, n)
    reco2d.Scale(float(normalization)/n)

    print(reco2d.Integral(), gen2d.Integral())

    #sys.exit()
    RESPONSE=response.Mresponse()
    singular=ROOT.TDecompSVD(RESPONSE)
    #print(singular.GetSig().Print())
    
    #unfold2d = ROOT.RooUnfoldInvert  (response, reco2d)
    #unfold2d =ROOT.RooUnfoldBinByBin  (response, reco2d)

    unfold2d = ROOT.RooUnfoldBayes  (response, reco2d, 4)
    unfold2d.HandleFakes(True)

    hUnf2d = unfold2d.Hunfold()
    hErr2d = unfold2d.Eunfold()
    
    reco = []
    gen = []
    unfold = []

    eec_reco = Proj2D_X(reco2d, eijbins2[1], eijbins2[-1], f"RECO_EEC")
    eec_gen = Proj2D_X(gen2d, eijbins2[1], eijbins2[-1], f"GEN_EEC")
    eec_unfold = Proj2D_X(hUnf2d, eijbins2[1], eijbins2[-1], f"UNFOLD_EEC")
    eec_reco.SetDirectory(0)
    eec_gen.SetDirectory(0)
    eec_unfold.SetDirectory(0)

    eec_reco.Scale(0.)
    eec_gen.Scale(0.)
    eec_unfold.Scale(0.)

    for i in range(len(eijbins2)-1):
        reco1d = Proj2D_X(reco2d, eijbins2[i], eijbins2[i+1], f"RECO_Eij_Bin{i}")
        gen1d = Proj2D_X(gen2d, eijbins2[i], eijbins2[i+1], f"GEN_Eij_Bin{i}")
        unfold1d = Proj2D_X(hUnf2d, eijbins2[i], eijbins2[i+1], f"UNFOLD_Eij_Bin{i}")
        reco1d.SetDirectory(0)
        gen1d.SetDirectory(0)
        unfold1d.SetDirectory(0)
        reco+=[reco1d]
        gen+=[gen1d]
        unfold+=[unfold1d]

        eec_reco.Add(reco1d)
        eec_gen.Add(gen1d)
        eec_unfold.Add(unfold1d)
    
    fout = ROOT.TFile(filenameout, 'recreate')
    fout.cd()
    for i in range(len(reco)):
        reco[i].Write()
        gen[i].Write()
        unfold[i].Write()

    hErr2d.Write()
    gen2d.Write()

    eec_reco.Write()
    eec_gen.Write()
    eec_unfold.Write()

    fout.Close()
        

    
