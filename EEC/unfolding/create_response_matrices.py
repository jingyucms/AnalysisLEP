import ROOT
import numpy as np
import sys
import os
import argparse
import math

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

sys.path.append(os.path.abspath("/afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python"))

from analysis_eec import *

#eijbins1 = [0, 0.01, 0.012589254117941675, 0.015848931924611134, 0.0199526231496888, 0.025118864315095794, 0.03162277660168379, 0.039810717055349734, 0.05011872336272725, 0.06309573444801933, 0.07943282347242814, 0.1, 0.12589254117941675, 0.15848931924611134, 10]

base = 2
start = math.log(0.0001, base)  # Log10 of the lower bound
end = math.log(0.3, base)       # Log10 of the upper bound
num_points = 35           # Number of points to generate

log_values = np.linspace(start, end, num_points)
eijbins1 = (base ** log_values)

eijbins1 = np.concatenate([[0], eijbins1])
eijbins1 = np.concatenate([eijbins1, [1]])

eijbins2 = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

rbins = calcBinEdge(0.001, np.pi/2, 100)

eijbins1 = np.array(eijbins1)
eijbins2 = np.array(eijbins2)
rbins = np.array(rbins)

matching_r = 0.05

def OpenFile(file_in,iodir):
    """  file_in -- Input file name
         iodir   -- 'r' readonly  'r+' read+write """
    try:
        ifile=open(file_in, iodir)
    except:
        print("Could not open file: ",file_in)
        sys.exit(1)
    return ifile

def ReadFilesFromList(infile):

    ifile=OpenFile(infile,'r')
    iline=0

    x = ifile.readline()

    filelist=[]
    while x != "":
        iline+=1
        filename=x.rstrip()

        if len(filename)>0 and filename[0] != "#":
            print(filename)
            filelist.append(filename)

        x = ifile.readline()

    return filelist

class MyResponse:

    def __init__(self, reco_tree, gen_tree):
        self._hists = {}
        self._resps = {}
        self._treco = reco_tree
        self._tgen = gen_tree
        self._evt_counter = 0

    def writeToFile(self, output):
        fout = ROOT.TFile(output, 'recreate')
        fout.SetCompressionLevel(9)
        fout.cd()
        for key in self._hists.keys():
            self._hists[key].Write()
        for key in self._resps.keys():
            self._resps[key].Write()
        fout.Close()

    def bookHistograms(self):

        hname = 'counter'
        self._hists[hname] = ROOT.TH1F(hname, hname, 1, 0, 1)

        # EEC hists
        hname = 'reco2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'gen2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'reco2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'gen2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'reco1d_eec'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)

        hname = 'gen1d_eec'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
        
        for i in range(len(eijbins1)-1):
            hname = f'reco_eij_r_bin1_{i}'
            self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
    
        for i in range(len(eijbins1)-1):
            hname = f'gen_eij_r_bin1_{i}'
            self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)

        for i in range(len(eijbins2)-1):
            hname = f'reco_eij_r_bin2_{i}'
            self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
    
        for i in range(len(eijbins2)-1):
            hname = f'gen_eij_r_bin2_{i}'
            self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
    
        hname = 'resp_r'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(rbins)-1, rbins)
        
        hname = 'resp_eij_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(eijbins1)-1, eijbins1, len(eijbins1)-1, eijbins1)

        hname = 'resp_eij_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(eijbins2)-1, eijbins2, len(eijbins2)-1, eijbins2)
    
        hname = 'fake_r'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
        
        hname = 'fake_eij_bin1'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(eijbins1)-1, eijbins1)

        hname = 'fake_eij_bin2'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(eijbins2)-1, eijbins2)
    
        hname = 'miss_r'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)

        hname = 'miss_eij_bin1'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(eijbins1)-1, eijbins1)
        
        hname = 'miss_eij_bin2'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(eijbins2)-1, eijbins2)

        # pt eta phi hists
        for c in ['match', 'fake', 'miss']:
            hname = f'reco_{c}_pt'
            self._hists[hname] = ROOT.TH1F(hname, hname, 450, 0, 45)

            hname = f'gen_{c}_pt'
            self._hists[hname] = ROOT.TH1F(hname, hname, 450, 0, 45)

            hname = f'reco_{c}_theta'
            self._hists[hname] = ROOT.TH1F(hname, hname, 30, 0.1, 3.1)

            hname = f'gen_{c}_theta'
            self._hists[hname] = ROOT.TH1F(hname, hname, 30, 0.1, 3.1)


    def bookResponseMatrices(self):

        respname = 'response2d_eij_r_bin1'
        self._resps[respname] = ROOT.RooUnfoldResponse(respname, respname)
        self._resps[respname].Setup(self._hists['reco2d_eij_r_bin1'], self._hists['gen2d_eij_r_bin1'])

        respname = 'response2d_eij_r_bin2'
        self._resps[respname] = ROOT.RooUnfoldResponse(respname, respname)
        self._resps[respname].Setup(self._hists['reco2d_eij_r_bin2'], self._hists['gen2d_eij_r_bin2'])
    
    def loop(self):

        nevt = self._treco.GetEntries()
        for n in range(nevt):
            self._evt_counter += 1

            self._hists['counter'].Fill(0.5)

            self._tgen.GetEntry(n)
            self._treco.GetEntry(n)
            
            
            ## gen event selection
            if self._tgen.passesSTheta < 0.5: continue

            ## reco event selection
            if self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5: continue

            ## gen track selection
            c_gen = np.array(self._tgen.charge)
            pt_gen = np.array(self._tgen.pt)
            theta_gen = np.array(self._tgen.theta)
            hp_gen = np.array(self._tgen.highPurity)
            sel_gen = (abs(c_gen) > 0.1) & (theta_gen < 2.795) & (theta_gen > 0.348) & (hp_gen > 0.5)

            c_gen = c_gen[sel_gen]
            E_gen = self._tgen.Energy
            px_gen = np.array(self._tgen.px)[sel_gen]
            py_gen = np.array(self._tgen.py)[sel_gen]
            pz_gen = np.array(self._tgen.pz)[sel_gen]
            m_gen = np.array(self._tgen.mass)[sel_gen]
            pt_gen = pt_gen[sel_gen]
            theta_gen = theta_gen[sel_gen]

            ## reco track selection     
            c_reco = np.array(self._treco.charge)
            pt_reco = np.array(self._treco.pt)
            theta_reco = np.array(self._treco.theta)
            pt_reco = np.array(self._treco.pt)
            hp_reco = np.array(self._treco.highPurity)
            sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
        
            c_reco = c_reco[sel_reco]
            E_reco = self._treco.Energy
            px_reco = np.array(self._treco.px)[sel_reco]
            py_reco = np.array(self._treco.py)[sel_reco]
            pz_reco = np.array(self._treco.pz)[sel_reco]
            m_reco = np.array(self._treco.mass)[sel_reco]
            pt_reco = pt_reco[sel_reco]
            theta_reco = theta_reco[sel_reco]
            
            nTrks_reco = len(px_reco)
            nTrks_gen = len(px_gen)
            
            dists = np.full((nTrks_reco, nTrks_gen), -1, 'd')

            # matching
            for i in range(nTrks_reco):
                for j in range(nTrks_gen):
                    pxi = px_reco[i]
                    pyi = py_reco[i]
                    pzi = pz_reco[i]
                    mi = m_reco[i]
        
                    pxj = px_gen[j]
                    pyj = py_gen[j]
                    pzj = pz_gen[j]
                    mj = m_gen[j]
        
                    cos_t = cos_theta(pxi, pyi, pzi, pxj, pyj, pzj)
                    dists[i, j] = c_reco[i]*c_gen[j]*theta(cos_t)
        
            dists[dists < 0] = 9999
        
            matched = [(i, j, dists[i, j]) for i in range(dists.shape[0]) for j in range(dists.shape[1]) if dists[i, j] < matching_r]
        
            matched = np.array(sorted(matched, key=lambda x: x[2]))
    
            matched = self.oneOnOneMatch(matched, 0)
            matched = self.oneOnOneMatch(matched, 1)
    
            matched_reco = matched[:, 0]
            matched_gen = matched[:, 1]
    
            miss = np.setxor1d(np.array(range(len(px_gen)), 'i'), np.array(matched_gen, 'i'))
    
            fake = np.setxor1d(np.array(range(len(px_reco)), 'i'), np.array(matched_reco, 'i'))

            # fill response matrices and histograms
            ## loop over matched
            n_match = 0
            for i in range(len(matched)):
                i_orig_reco = int(matched[i][0])
                i_orig_gen = int(matched[i][1])
                self._hists['reco_match_pt'].Fill(pt_reco[i_orig_reco])
                self._hists['gen_match_pt'].Fill(pt_gen[i_orig_gen])
                self._hists['reco_match_theta'].Fill(theta_reco[i_orig_reco])
                self._hists['gen_match_theta'].Fill(theta_gen[i_orig_gen])
                
                for j in range(len(matched)):
                    if i>=j: continue
                    j_orig_reco = int(matched[j][0])
                    j_orig_gen = int(matched[j][1])
                    
                    Eij_reco = E_ij(px_reco[i_orig_reco], py_reco[i_orig_reco], pz_reco[i_orig_reco], m_reco[i_orig_reco], px_reco[j_orig_reco], py_reco[j_orig_reco], pz_reco[j_orig_reco], m_reco[j_orig_reco])/E_reco**2
                    Eij_gen = E_ij(px_gen[i_orig_gen], py_gen[i_orig_gen], pz_gen[i_orig_gen], m_gen[i_orig_gen], px_gen[j_orig_gen], py_gen[j_orig_gen], pz_gen[j_orig_gen], m_gen[j_orig_gen])/E_gen**2
                    cr_reco = cos_theta(px_reco[i_orig_reco], py_reco[i_orig_reco], pz_reco[i_orig_reco], px_reco[j_orig_reco], py_reco[j_orig_reco], pz_reco[j_orig_reco])
                    cr_gen = cos_theta(px_gen[i_orig_gen], py_gen[i_orig_gen], pz_gen[i_orig_gen], px_gen[j_orig_gen], py_gen[j_orig_gen], pz_gen[j_orig_gen])
                    r_reco = theta(cr_reco)
                    r_gen = theta(cr_gen)
                    self._hists['resp_eij_bin1'].Fill(Eij_reco, Eij_gen)
                    self._hists['resp_eij_bin2'].Fill(Eij_reco, Eij_gen)
                    self._hists['resp_r'].Fill(r_reco, r_gen)
    
                    i_plot_reco_bin1 = self.findBin(eijbins1, Eij_reco)
                    self._hists[f'reco_eij_r_bin1_{i_plot_reco_bin1}'].Fill(r_reco, Eij_reco)
                    i_plot_gen_bin1 = self.findBin(eijbins1, Eij_gen)
                    self._hists[f'gen_eij_r_bin1_{i_plot_gen_bin1}'].Fill(r_gen, Eij_gen)

                    i_plot_reco_bin2 = self.findBin(eijbins2, Eij_reco)
                    self._hists[f'reco_eij_r_bin2_{i_plot_reco_bin2}'].Fill(r_reco, Eij_reco)
                    i_plot_gen_bin2 = self.findBin(eijbins2, Eij_gen)
                    self._hists[f'gen_eij_r_bin2_{i_plot_gen_bin2}'].Fill(r_gen, Eij_gen)

                    self._hists['reco2d_eij_r_bin1'].Fill(r_reco, Eij_reco)
                    self._hists['gen2d_eij_r_bin1'].Fill(r_gen, Eij_gen)

                    self._hists['reco2d_eij_r_bin2'].Fill(r_reco, Eij_reco)
                    self._hists['gen2d_eij_r_bin2'].Fill(r_gen, Eij_gen)

                    self._hists['reco1d_eec'].Fill(r_reco, Eij_reco)
                    self._hists['gen1d_eec'].Fill(r_gen, Eij_gen)

                    self._resps['response2d_eij_r_bin1'].Fill(r_reco, Eij_reco, r_gen, Eij_gen)
                    self._resps['response2d_eij_r_bin2'].Fill(r_reco, Eij_reco, r_gen, Eij_gen)
                    n_match+=1

            ## loop over miss
            n_miss = 0
            for i in range(len(miss)):
                i_orig = miss[i]
                self._hists['gen_miss_pt'].Fill(pt_gen[i_orig])
                self._hists['gen_miss_theta'].Fill(theta_gen[i_orig])
                for j in range(len(miss)):
                    if i>=j: continue
                    j_orig = miss[j]
                    Eij = E_ij(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], m_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig], m_gen[j_orig])/E_gen**2
                    cr = cos_theta(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig])
                    r = theta(cr)
                    self._hists['miss_r'].Fill(r, Eij)
                    self._hists['miss_eij_bin1'].Fill(Eij)
                    self._hists['gen2d_eij_r_bin1'].Fill(r, Eij)
                    self._hists['miss_eij_bin2'].Fill(Eij)
                    self._hists['gen2d_eij_r_bin2'].Fill(r, Eij)
                    self._hists['gen1d_eec'].Fill(r, Eij)
                    self._resps['response2d_eij_r_bin1'].Miss(r, Eij)
                    self._resps['response2d_eij_r_bin2'].Miss(r, Eij)
                    n_miss+=1

                for j in range(len(matched)):
                    i_orig = miss[i]
                    j_orig = int(matched[j][1])
                    Eij = E_ij(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], m_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig], m_gen[j_orig])/E_gen**2
                    cr = cos_theta(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig])
                    r = theta(cr)
                    self._hists['miss_r'].Fill(r, Eij)
                    self._hists['gen1d_eec'].Fill(r, Eij)
                    self._hists['miss_eij_bin1'].Fill(Eij)
                    self._hists['miss_eij_bin2'].Fill(Eij)
                    self._hists['gen2d_eij_r_bin1'].Fill(r, Eij)
                    self._hists['gen2d_eij_r_bin2'].Fill(r, Eij)
                    self._resps['response2d_eij_r_bin1'].Miss(r, Eij)
                    self._resps['response2d_eij_r_bin2'].Miss(r, Eij)
                    n_miss+=1

            ## loop over fake
            n_fake = 0
            for i in range(len(fake)):
                i_orig = fake[i]
                self._hists['reco_fake_pt'].Fill(pt_reco[i_orig])
                self._hists['reco_fake_theta'].Fill(theta_reco[i_orig])
                for j in range(len(fake)):
                    if i>=j: continue
                    j_orig = fake[j]
                    Eij = E_ij(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], m_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig], m_reco[j_orig])/E_reco**2
                    cr = cos_theta(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig])
                    r = theta(cr)
                    self._hists['fake_r'].Fill(r, Eij)
                    self._hists['fake_eij_bin1'].Fill(Eij)
                    self._hists['reco2d_eij_r_bin1'].Fill(r, Eij)
                    self._hists['fake_eij_bin2'].Fill(Eij)
                    self._hists['reco2d_eij_r_bin2'].Fill(r, Eij)
                    self._hists['reco1d_eec'].Fill(r, Eij)
                    self._resps['response2d_eij_r_bin1'].Fake(r, Eij)
                    self._resps['response2d_eij_r_bin2'].Fake(r, Eij)
                    n_fake+=1
                    
                for j in range(len(matched)):
                    i_orig = fake[i]
                    j_orig = int(matched[j][0])
                    Eij = E_ij(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], m_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig], m_reco[j_orig])/E_reco**2
                    cr = cos_theta(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig])
                    r = theta(cr)
                    self._hists['fake_r'].Fill(r, Eij)
                    self._hists['fake_eij_bin1'].Fill(Eij)
                    self._hists['reco2d_eij_r_bin1'].Fill(r, Eij)
                    self._hists['fake_eij_bin2'].Fill(Eij)
                    self._hists['reco2d_eij_r_bin2'].Fill(r, Eij)
                    self._hists['reco1d_eec'].Fill(r, Eij)
                    self._resps['response2d_eij_r_bin1'].Fake(r, Eij)
                    self._resps['response2d_eij_r_bin2'].Fake(r, Eij)
                    n_fake+=1

    def oneOnOneMatch(self, matched, flag):
        seen = set()
        oneOnOne = []
        
        for value in matched:
            if value[flag] not in seen:
                seen.add(value[flag])
                oneOnOne.append(value)
        
        return np.array(oneOnOne)

    def findBin(self, bin_edges, value):
        bin_index = np.digitize(value, bin_edges) - 1

        return bin_index

    def normalize(self):
        self._hists['reco1d_eec'].Scale(1./self._evt_counter)
        self._hists['reco1d_eec']=normalizeByBinWidth(self._hists['reco1d_eec'])
        
        self._hists['gen1d_eec'].Scale(1./self._evt_counter)
        self._hists['gen1d_eec']=normalizeByBinWidth(self._hists['gen1d_eec'])
    

if __name__ == "__main__":


    filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/LEP1MC1994_recons_aftercut-001.root'
    filenameout = "response.root"

    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    treco = 't'
    tgen = 'tgen'

    t_reco = ROOT.TChain(treco)
    t_gen = ROOT.TChain(tgen)

    print("Reading input from: ",args.infiles)
    InputRootFiles=[]
    if args.infiles.find(".root")>-1:
        InputRootFiles.append(args.infiles)
    else:
        ## read from list
        InputRootFiles=ReadFilesFromList(args.infiles)
    
    for f in InputRootFiles:
        t_reco.Add(f)
        t_gen.Add(f)

    fnameout = args.outfile

    response = MyResponse(t_reco, t_gen)
    response.bookHistograms()
    response.bookResponseMatrices()
    response.loop()
    response.normalize()
    response.writeToFile(fnameout)
