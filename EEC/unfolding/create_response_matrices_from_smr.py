import ROOT
import numpy as np
import sys
import os
import argparse
import math
from scipy.optimize import linear_sum_assignment
import json

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

sys.path.append(os.path.abspath("/afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python"))

def cos_theta(p1x, p1y, p1z, p2x, p2y, p2z):
    num = p1x*p2x + p1y*p2y + p1z*p2z
    den = math.sqrt(p1x**2+p1y**2+p1z**2)*math.sqrt(p2x**2+p2y**2+p2z**2)
    cos_theta = num/den
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return cos_theta

def theta(ct):
    theta = math.acos(ct)
    return theta

def E(p1x, p1y, p1z, m1):
    return math.sqrt(p1x**2+p1y**2+p1z**2+m1**2)

def E_ij(p1x, p1y, p1z, m1, p2x, p2y, p2z, m2):
    E_i = E(p1x, p1y, p1z, m1)
    E_j = E(p2x, p2y, p2z, m2)
    return E_i*E_j

def calcBinEdge(low, high, nbins):   
    xlow = np.log10(low)
    xhigh = np.log10(high)
    width = (xhigh-xlow)/nbins

    bins=[]
    for i in range(nbins+1):
        val = pow(10, xlow + i * width)
        bins += [val]
    
    newbins = [2*high - b for b in bins]
    newbins = newbins[::-1]
    del newbins[0]
    bin_edge = bins + newbins

    return bin_edge

def normalizeByBinWidth(h):
    for b in range(h.GetNbinsX()):
        h.SetBinContent(b+1, h.GetBinContent(b+1)/h.GetBinWidth(b+1))
        h.SetBinError(b+1, h.GetBinError(b+1)/h.GetBinWidth(b+1))

    return h

def matching_metric_aleph(theta1, phi1, e1, theta2, phi2, e2):
    delta_theta = abs(theta1-theta2)
    delta_phi = abs(phi1-phi2)
    delta_e = abs(e1-e2)
    mean_e = 0.5*(e1+e2)
    
    scale_factor_theta = 2.8  
    scale_factor_phi = 2.3    
    scale_factor_e = 1

    sigma_delta = 25e-6 + 95e-6 / mean_e
    inner_vertex_radius = 6e-2

    sigma_theta = (sigma_delta / inner_vertex_radius) * scale_factor_theta
    sigma_phi = (sigma_delta / inner_vertex_radius) * scale_factor_phi
    sigma_e = np.sqrt((6e-4 * mean_e)**2 + 0.005**2) * mean_e * scale_factor_e

    chi_theta = delta_theta / sigma_theta
    chi_phi = delta_phi / sigma_phi
    chi_e = delta_e / sigma_e

    chi2_metric = chi_theta**2 + chi_phi**2 + chi_e**2

    return chi2_metric

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

def smr_pdf(x, par):
    
    mu = par[0]
    sigma = par[1]
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    return norm * math.exp(-0.5 * ((x[0] - mu) / sigma)**2)

base = 2
start = math.log(0.0001, base)  # Log10 of the lower bound
end = math.log(0.3, base)       # Log10 of the upper bound
num_points = 35           # Number of points to generate

log_values = np.linspace(start, end, num_points)
eijbins1 = (base ** log_values)

eijbins1 = np.concatenate([[0], eijbins1])
eijbins1 = np.concatenate([eijbins1, [1]])

eijbins2 = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

rbins = calcBinEdge(0.002, np.pi/2, 100)

eijbins1 = np.array(eijbins1)
eijbins2 = np.array(eijbins2)
rbins = np.array(rbins)

doAngular = False

if doAngular:
    matching_r = 0.05
    #matching_r = 3
else:
    matching_r = 10000

class MyResponseSMR:

    def __init__(self, gen_tree):
        self._hists = {}
        self._resps = {}
        self._tgen = gen_tree
        self._evt_counter = 0
        self._pdfs = []

    def writeToFile(self, output):
        fout = ROOT.TFile(output, 'recreate')
        fout.SetCompressionLevel(9)
        fout.cd()
        for key in self._hists.keys():
            self._hists[key].Write()
        #for key in self._resps.keys():
        #    self._resps[key].Write()
        #fout.Close()

    def bookHistograms(self):

        hname = 'counter'
        self._hists[hname] = ROOT.TH1F(hname, hname, 1, 0, 1)

        # EEC hists
        hname = 'smr2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'gen2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'smr2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'gen2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'smr1d_eec'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)

        hname = 'gen1d_eec'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(rbins)-1, rbins)
    
        hname = 'resp_r'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(rbins)-1, rbins, len(rbins)-1, rbins)
        
        hname = 'resp_eij_bin1'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(eijbins1)-1, eijbins1, len(eijbins1)-1, eijbins1)

        hname = 'resp_eij_bin2'
        self._hists[hname] = ROOT.TH2F(hname, hname, len(eijbins2)-1, eijbins2, len(eijbins2)-1, eijbins2)

        # pt eta phi hists
        for c in ['match']:
            hname = f'smr_{c}_pt'
            self._hists[hname] = ROOT.TH1F(hname, hname, 450, 0, 45)

            hname = f'gen_{c}_pt'
            self._hists[hname] = ROOT.TH1F(hname, hname, 450, 0, 45)

            hname = f'smr_{c}_theta'
            self._hists[hname] = ROOT.TH1F(hname, hname, 30, 0.1, 3.1)

            hname = f'gen_{c}_theta'
            self._hists[hname] = ROOT.TH1F(hname, hname, 30, 0.1, 3.1)


    def bookResponseMatrices(self):

        respname = 'response2d_eij_r_bin1'
        self._resps[respname] = ROOT.RooUnfoldResponse(respname, respname)
        self._resps[respname].Setup(self._hists['smr2d_eij_r_bin1'], self._hists['gen2d_eij_r_bin1'])

        respname = 'response2d_eij_r_bin2'
        self._resps[respname] = ROOT.RooUnfoldResponse(respname, respname)
        self._resps[respname].Setup(self._hists['smr2d_eij_r_bin2'], self._hists['gen2d_eij_r_bin2'])
    
    def loop(self):

        nevt = self._tgen.GetEntries()
        for n in range(nevt):
            self._evt_counter += 1

            self._hists['counter'].Fill(0.5)

            self._tgen.GetEntry(n)
            
            ## gen event selection
            if self._tgen.passesSTheta < 0.5: continue

            E = self._tgen.Energy

            ## gen track selection
            c_gen = np.array(self._tgen.charge)
            pt_gen = np.array(self._tgen.pt)
            theta_gen = np.array(self._tgen.theta)
            hp_gen = np.array(self._tgen.highPurity)
            #sel_gen = (abs(c_gen) > 0.1) & (pt_gen > 0.2) & (theta_gen < 2.795) & (theta_gen > 0.348) & (hp_gen > 0.5)
            sel_gen = (abs(c_gen) > 0.1) & (hp_gen > 0.5)

            c_gen = c_gen[sel_gen]
            E_gen = self._tgen.Energy
            px_gen = np.array(self._tgen.px)[sel_gen]
            py_gen = np.array(self._tgen.py)[sel_gen]
            pz_gen = np.array(self._tgen.pz)[sel_gen]
            m_gen = np.array(self._tgen.mass)[sel_gen]
            pt_gen = pt_gen[sel_gen]
            theta_gen = theta_gen[sel_gen]
            phi_gen = np.array(self._tgen.phi)[sel_gen]

            nTrks = len(px_gen)

            e_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2 + m_gen**2)

            smrbins = [0, 2, 4, 7, 10, 15, 20, 30, 40, 200]

            e_reco = np.zeros(nTrks, 'd')

            for i, e in enumerate(e_gen):
                ibin = self.findBin(smrbins, e)
                factor = self._pdfs[ibin].GetRandom()
                e_reco[i] = e*factor


            factor_theta = np.random.normal(loc=1.0, scale=0.001, size=nTrks)
            factor_phi = np.random.normal(loc=1.0, scale=0.001, size=nTrks)

            theta_reco = theta_gen * factor_theta
            phi_reco = phi_gen * factor_phi

            m_reco = m_gen

            p_reco = np.sqrt(e_reco**2-m_reco**2)

            px_reco = p_reco * np.sin(theta_reco) * np.cos(phi_reco)
            py_reco = p_reco * np.sin(theta_reco) * np.sin(phi_reco)
            pz_reco = p_reco * np.cos(theta_reco)

            dists = np.full((nTrks, nTrks), -1, 'd')

            # matching
            for i in range(nTrks):
                for j in range(nTrks):
                    pxi = px_reco[i]
                    pyi = py_reco[i]
                    pzi = pz_reco[i]
                    mi = m_reco[i]
                    itheta = theta_reco[i]
                    iphi = phi_reco[i]
        
                    pxj = px_gen[j]
                    pyj = py_gen[j]
                    pzj = pz_gen[j]
                    mj = m_gen[j]
                    jtheta = theta_gen[j]
                    jphi = phi_gen[j]

                    ei = e_reco[i]
                    ej = e_gen[i]
                    dists[i, j] = matching_metric_aleph(itheta, iphi, ei, jtheta, jphi, ej)
        
            dists[dists < 0] = 99999
    

            matched_reco, matched_gen = linear_sum_assignment(dists)
            matched = np.column_stack((matched_reco, matched_gen))
    
            miss = np.setxor1d(np.array(range(len(px_gen)), 'i'), np.array(matched_gen, 'i'))
            fake = np.setxor1d(np.array(range(len(px_reco)), 'i'), np.array(matched_reco, 'i'))

            #print(len(matched), len(miss), len(fake))
        
            # fill response matrices and histograms
            ## loop over matched
            n_match = 0
            for i in range(len(matched)):
                i_orig_reco = int(matched[i][0])
                i_orig_gen = int(matched[i][1])
                self._hists['smr_match_pt'].Fill(e_reco[i_orig_reco])
                self._hists['gen_match_pt'].Fill(e_gen[i_orig_gen])
                self._hists['smr_match_theta'].Fill(theta_reco[i_orig_reco])
                self._hists['gen_match_theta'].Fill(theta_gen[i_orig_gen])
                
                for j in range(len(matched)):
                    if i>=j: continue
                    j_orig_reco = int(matched[j][0])
                    j_orig_gen = int(matched[j][1])

                    Eij_reco = e_reco[i_orig_reco]*e_reco[j_orig_reco]/E**2
                    Eij_gen = e_gen[i_orig_gen]*e_gen[j_orig_gen]/E**2
                    
                    cr_reco = cos_theta(px_reco[i_orig_reco], py_reco[i_orig_reco], pz_reco[i_orig_reco], px_reco[j_orig_reco], py_reco[j_orig_reco], pz_reco[j_orig_reco])
                    cr_gen = cos_theta(px_gen[i_orig_gen], py_gen[i_orig_gen], pz_gen[i_orig_gen], px_gen[j_orig_gen], py_gen[j_orig_gen], pz_gen[j_orig_gen])
                    r_reco = theta(cr_reco)
                    r_gen = theta(cr_gen)
                    self._hists['resp_eij_bin1'].Fill(Eij_reco, Eij_gen)
                    self._hists['resp_eij_bin2'].Fill(Eij_reco, Eij_gen)
                    self._hists['resp_r'].Fill(r_reco, r_gen)

                    self._hists['smr2d_eij_r_bin1'].Fill(r_reco, Eij_reco)
                    self._hists['gen2d_eij_r_bin1'].Fill(r_gen, Eij_gen)

                    self._hists['smr2d_eij_r_bin2'].Fill(r_reco, Eij_reco)
                    self._hists['gen2d_eij_r_bin2'].Fill(r_gen, Eij_gen)

                    self._hists['smr1d_eec'].Fill(r_reco, Eij_reco)
                    self._hists['gen1d_eec'].Fill(r_gen, Eij_gen)

                    self._resps['response2d_eij_r_bin1'].Fill(r_reco, Eij_reco, r_gen, Eij_gen)
                    self._resps['response2d_eij_r_bin2'].Fill(r_reco, Eij_reco, r_gen, Eij_gen)
                    n_match+=1

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

    def loadSMRParams(self, fparams):
        with open(fparams, "r") as file:
            params = json.load(file)
             
        for i, param in enumerate(params):
            func = ROOT.TF1(f"smr_e{i}", smr_pdf, 0.2, 1.8, 2)
            func.SetParameters(param['mean'], param['sigma'])
            self._pdfs.append(func)

if __name__ == "__main__":


    filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/LEP1MC1994_recons_aftercut-001.root'
    filenameout = "smeared_response.root"

    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    tgen = 't'

    t_gen = ROOT.TChain(tgen)

    print("Reading input from: ",args.infiles)
    InputRootFiles=[]
    if args.infiles.find(".root")>-1:
        InputRootFiles.append(args.infiles)
    else:
        ## read from list
        InputRootFiles=ReadFilesFromList(args.infiles)
    
    for f in InputRootFiles:
        t_gen.Add(f)

    fnameout = args.outfile

    response = MyResponseSMR(t_gen)
    response.loadSMRParams("fit_results.json")
    response.bookHistograms()
    response.bookResponseMatrices()
    response.loop()
    #response.normalize()
    response.writeToFile(fnameout)
