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

if __name__ == '__main__':

    filenamein = "response_LEP1MC1994_v4.root"

    filenameout = "unfolded_v4_bin2.root"

    fin = ROOT.TFile.Open(filenamein)
    
    response = fin.Get("response2d_eij_r_bin2")

    reco2d = fin.Get("reco2d_eij_r_bin2")

    gen2d = fin.Get("gen2d_eij_r_bin2")

    normalization = fin.Get("counter").GetBinContent(1)

    RESPONSE=response.Mresponse()
    singular=ROOT.TDecompSVD(RESPONSE)
    #print(singular.GetSig().Print())
    
    
    unfold2d = ROOT.RooUnfoldBayes  (response, reco2d, 4)
    unfold2d.HandleFakes(True)
    #unfold2d = ROOT.RooUnfoldInvert  (response, reco2d)
    #unfold2d =ROOT.RooUnfoldBinByBin  (response, reco2d)

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
        #if not i == 0:
##         bincenter = (eijbins1[i]+eijbins1[i+1])/2
##         reco1d.Scale(bincenter)
##         gen1d.Scale(bincenter)
##         unfold1d.Scale(bincenter)
        eec_reco.Add(reco1d)
        eec_gen.Add(gen1d)
        eec_unfold.Add(unfold1d)
            
##     eec_reco.Scale(1./normalization)
##     eec_gen.Scale(1./normalization)
##     eec_unfold.Scale(1./normalization)
## 
##     eec_reco = normalizeByBinWidth(eec_reco)
##     eec_gen = normalizeByBinWidth(eec_gen)
##     eec_unfold = normalizeByBinWidth(eec_unfold)
    
    fout = ROOT.TFile(filenameout, 'recreate')
    fout.cd()
    for i in range(len(reco)):
        reco[i].Write()
        gen[i].Write()
        unfold[i].Write()

    hErr2d.Write()

    eec_reco.Write()
    eec_gen.Write()
    eec_unfold.Write()

    fout.Close()
        

    
