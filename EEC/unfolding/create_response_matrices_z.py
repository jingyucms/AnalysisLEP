import ROOT
import numpy as np
import sys
import os
import argparse
import math
from scipy.optimize import linear_sum_assignment

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

sys.path.append(os.path.abspath("/afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python"))

def calcAngle(n1, n2):
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    
    theta = np.arccos(cos_theta)
    return theta

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

base = 2
start = math.log(0.0001, base)  # Log10 of the lower bound
end = math.log(0.3, base)       # Log10 of the upper bound
num_points = 35           # Number of points to generate

log_values = np.linspace(start, end, num_points)
eijbins1 = (base ** log_values)

eijbins1 = np.concatenate([[0], eijbins1])
eijbins1 = np.concatenate([eijbins1, [1]])

eijbins2 = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

rbins = calcBinEdge(0.000001, 0.5, 100)

eijbins1 = np.array(eijbins1)
eijbins2 = np.array(eijbins2)
rbins = np.array(rbins)

doAngular = True

if doAngular:
    matching_r = 0.05
    #matching_r = 3
else:
    matching_r = 10000

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
        self._hists[hname] = ROOT.TH1D(hname, hname, 2, 0, 2)

        # EEC hists
        hname = 'reco2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'gen2d_eij_r_bin1'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(rbins)-1, rbins, len(eijbins1)-1, eijbins1)

        hname = 'reco2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'gen2d_eij_r_bin2'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(rbins)-1, rbins, len(eijbins2)-1, eijbins2)

        hname = 'reco1d_eec'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)

        hname = 'gen1d_eec'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)
        
        for i in range(len(eijbins1)-1):
            hname = f'reco_eij_r_bin1_{i}'
            self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)
    
        for i in range(len(eijbins1)-1):
            hname = f'gen_eij_r_bin1_{i}'
            self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)

        for i in range(len(eijbins2)-1):
            hname = f'reco_eij_r_bin2_{i}'
            self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)
    
        for i in range(len(eijbins2)-1):
            hname = f'gen_eij_r_bin2_{i}'
            self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)
    
        hname = 'resp_r'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(rbins)-1, rbins, len(rbins)-1, rbins)
        
        hname = 'resp_eij_bin1'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(eijbins1)-1, eijbins1, len(eijbins1)-1, eijbins1)

        hname = 'resp_eij_bin2'
        self._hists[hname] = ROOT.TH2D(hname, hname, len(eijbins2)-1, eijbins2, len(eijbins2)-1, eijbins2)
    
        hname = 'fake_r'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)
        
        hname = 'fake_eij_bin1'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(eijbins1)-1, eijbins1)

        hname = 'fake_eij_bin2'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(eijbins2)-1, eijbins2)
    
        hname = 'miss_r'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(rbins)-1, rbins)

        hname = 'miss_eij_bin1'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(eijbins1)-1, eijbins1)
        
        hname = 'miss_eij_bin2'
        self._hists[hname] = ROOT.TH1D(hname, hname, len(eijbins2)-1, eijbins2)

        # pt eta phi hists
        for c in ['match', 'fake', 'miss']:
            hname = f'reco_{c}_pt'
            self._hists[hname] = ROOT.TH1D(hname, hname, 450, 0, 45)

            hname = f'gen_{c}_pt'
            self._hists[hname] = ROOT.TH1D(hname, hname, 450, 0, 45)

            hname = f'reco_{c}_theta'
            self._hists[hname] = ROOT.TH1D(hname, hname, 30, 0.1, 3.1)

            hname = f'gen_{c}_theta'
            self._hists[hname] = ROOT.TH1D(hname, hname, 30, 0.1, 3.1)


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
            #if self._tgen.passesSTheta < 0.5: continue

            ## reco event selection
            if self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5: continue

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

            e_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2 + m_gen**2)

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
            phi_reco = np.array(self._treco.phi)[sel_reco]

            e_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2 + m_reco**2)
            if np.sum(e_reco) > 200 or np.sum(e_gen) > 200: continue

            self._hists['counter'].Fill(1.5)
            
            nTrks_reco = len(px_reco)
            nTrks_gen = len(px_gen)
            
            # --- Vectorized matching cost matrix ---
            if doAngular:
                norm_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2)
                norm_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2)
                dot = np.outer(px_reco, px_gen) + np.outer(py_reco, py_gen) + np.outer(pz_reco, pz_gen)
                cos_t = dot / (np.outer(norm_reco, norm_gen) + 1e-12)
                cos_t = np.clip(cos_t, -1.0, 1.0)
                r_vals = np.arccos(cos_t)
                dists = (c_reco[:, None] * c_gen[None, :]) * r_vals
            else:
                dtheta = np.abs(theta_reco[:, None] - theta_gen[None, :])
                dphi = np.abs(phi_reco[:, None] - phi_gen[None, :])
                de = np.abs(e_reco[:, None] - e_gen[None, :])
                mean_e = 0.5 * (e_reco[:, None] + e_gen[None, :])
                sigma_delta = 25e-6 + 95e-6 / (mean_e + 1e-12)
                inner_radius = 6e-2
                sigma_theta = (sigma_delta / inner_radius) * 2.8
                sigma_phi = (sigma_delta / inner_radius) * 2.3
                sigma_e = np.sqrt((6e-4 * mean_e)**2 + 0.005**2) * mean_e
                chi_theta = dtheta / (sigma_theta + 1e-12)
                chi_phi = dphi / (sigma_phi + 1e-12)
                chi_e = de / (sigma_e + 1e-12)
                dists = chi_theta**2 + chi_phi**2 + chi_e**2
        
            dists[dists < 0] = 99999

            if doAngular:
                # For angular matching, select pairs having cost below threshold
                matched = np.array([
                    (i, j, dists[i, j])
                    for i in range(dists.shape[0])
                    for j in range(dists.shape[1])
                    if dists[i, j] < matching_r
                ])

                matched = np.array(sorted(matched, key=lambda x: x[2]))
    
                matched = self.oneOnOneMatch(matched, 0)
                matched = self.oneOnOneMatch(matched, 1)
    
                matched_reco = matched[:, 0]
                matched_gen = matched[:, 1]
                
            else:
                matched_reco, matched_gen = linear_sum_assignment(dists)
                matched = np.column_stack((matched_reco, matched_gen))
    
            miss = np.setxor1d(np.array(range(len(px_gen)), 'i'), np.array(matched_gen, 'i'))
            fake = np.setxor1d(np.array(range(len(px_reco)), 'i'), np.array(matched_reco, 'i'))
        
            # fill response matrices and histograms
            ## loop over matched
            n_match = 0
            for i in range(len(matched)):
                i_orig_reco = int(matched[i][0])
                i_orig_gen = int(matched[i][1])
                self._hists['reco_match_pt'].Fill(e_reco[i_orig_reco])
                self._hists['gen_match_pt'].Fill(e_gen[i_orig_gen])
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
                    #r_reco = theta(cr_reco)
                    #r_gen = theta(cr_gen)
                    r_reco = (1-cr_reco)/2.
                    r_gen = (1-cr_gen)/2.
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
                    r = (1-cr)/2.
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
                    j_orig = int(matched[j][1])
                    Eij = E_ij(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], m_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig], m_gen[j_orig])/E_gen**2
                    cr = cos_theta(px_gen[i_orig], py_gen[i_orig], pz_gen[i_orig], px_gen[j_orig], py_gen[j_orig], pz_gen[j_orig])
                    r = (1-cr)/2.
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
                    r = (1-cr)/2.
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
                    j_orig = int(matched[j][0])
                    Eij = E_ij(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], m_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig], m_reco[j_orig])/E_reco**2
                    cr = cos_theta(px_reco[i_orig], py_reco[i_orig], pz_reco[i_orig], px_reco[j_orig], py_reco[j_orig], pz_reco[j_orig])
                    r = (1-cr)/2.
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
    #response.normalize()
    response.writeToFile(fnameout)
