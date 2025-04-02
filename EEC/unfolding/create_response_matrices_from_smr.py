import ROOT
import numpy as np
import sys
import os
import argparse
import math
from scipy.optimize import linear_sum_assignment
import json
import time

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

sys.path.append(os.path.abspath("/afs/cern.ch/user/z/zhangj/private/ALEPH/CMSSW_14_1_5/src/AnalysisLEP/EEC/python"))

def calculate_sphericity(px, py, pz):
    """
    Calculate the sphericity matrix and perform eigen-decomposition to obtain
    eigenvalues and eigenvectors for the standard (nonlinear) sphericity (r=2).
    
    In this version, we assume that all particles are used (i.e. no particle flag filtering),
    and the weight factor is unity (since (p2)^(0) = 1).
    
    Additionally, the function returns the cosine of the polar angle (theta) for the
    first eigenvector. Since eigenvectors are normalized, the cosine of theta is simply
    the z-component of the eigenvector.
    
    Parameters:
      px, py, pz : iterables of floats
          The momentum components for each particle.
    
    Returns:
      A dictionary containing:
        "eigenvalues"  : NumPy array of eigenvalues (sorted in ascending order).
        "eigenvectors" : NumPy array of the corresponding eigenvectors as columns.
        "cos_theta_v1" : Cosine of the polar angle for the first eigenvector.
        "type"         : A string, "nonlinear" (since r is assumed to be 2).
    """
    def p2(px_val, py_val, pz_val):
        return px_val * px_val + py_val * py_val + pz_val * pz_val

    n = len(px)
    # Initialize a 3x3 matrix for the sphericity calculation.
    m = np.zeros((3, 3), dtype=float)
    norm = 0.0

    # Loop over each particle.
    for i in range(n):
        p2_val = p2(px[i], py[i], pz[i])
        factor = 1.0  # For r = 2, the exponent is (2-2)/2 = 0, so factor is 1.
        m[0, 0] += px[i] * px[i] * factor
        m[1, 1] += py[i] * py[i] * factor
        m[2, 2] += pz[i] * pz[i] * factor
        m[1, 0] += px[i] * py[i] * factor
        m[2, 0] += px[i] * pz[i] * factor
        m[1, 2] += py[i] * pz[i] * factor
        # Accumulate norm: for r = 2, it is the sum of p2 values.
        norm += p2_val

    # Normalize the matrix.
    if (norm == 0): print(px, py, pz)
    m = m / norm

    # Symmetrize the matrix explicitly.
    m[0, 1] = m[1, 0]
    m[0, 2] = m[2, 0]
    m[2, 1] = m[1, 2]

    # Compute the eigenvalues and eigenvectors.
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(m)

    # The first eigenvector is taken as the first column in eigenvectors.
    # Since the eigenvectors are normalized, the cosine theta of the first eigenvector
    # is simply its z-component (the third component).
    cos_theta_v1 = eigenvectors[2, 2]

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,  # Each column corresponds to an eigenvector.
        "cos_theta_v1": cos_theta_v1,
        "type": "nonlinear"
    }

def double_sided_crystal_ball(x, par):
    """
    Double-Sided Crystal Ball Function.
    
    Parameters:
      x      : Array-like, expected to be of length 1 (e.g. x[0]).
      par    : List or array of parameters.
               par[0] = N (normalization)
               par[1] = mu (mean)
               par[2] = sigma (width)
               par[3] = alpha_l (left threshold, positive)
               par[4] = n_l (left tail exponent)
               par[6] = alpha_r (right threshold, positive)
               par[7] = n_r (right tail exponent)
    
    Returns:
      The function evaluated at x[0] (a float).
    """
    
    # Unpack parameters
    N     = par[0]
    mu    = par[1]
    sigma = par[2]
    alpha_l = par[3]
    n_l     = par[4]
    alpha_r = par[6]
    n_r     = par[7]
    
    # Use only the first element of x
    x_val = x[0]
    t = (x_val - mu) / sigma
    
    # Evaluate based on region
    if t < -alpha_l:
        A_l = (n_l / abs(alpha_l))**n_l * math.exp(-0.5 * alpha_l**2)
        B_l = n_l / abs(alpha_l) - abs(alpha_l)
        result = N * A_l * (B_l - t)**(-n_l)
    elif t > alpha_r:
        A_r = (n_r / abs(alpha_r))**n_r * math.exp(-0.5 * alpha_r**2)
        B_r = n_r / abs(alpha_r) - abs(alpha_r)
        result = N * A_r * (B_r + t)**(-n_r)
    else:
        result = N * math.exp(-0.5 * t**2)
    
    return float(result)

def gaussian(x, par):
    
    mu = par[0]
    sigma = par[1]
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    return norm * math.exp(-0.5 * ((x[0] - mu) / sigma)**2)

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
        self._hists[hname] = ROOT.TH1F(hname, hname, 2, 0, 2)

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

            nTrks_gen = len(px_gen)

            e_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2 + m_gen**2)

            if np.sum(e_gen) > 200: continue

            smrbins = [0, 2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 200]

            e_reco = np.zeros(nTrks_gen, 'd')

            for i, e in enumerate(e_gen):
                ibin = self.findBin(smrbins, e)
                #print(ibin, self._pdfs[ibin])
                factor = self._pdfs[ibin].GetRandom()
                e_reco[i] = e*factor
                #if e_reco[i] < 2.1 and e_reco[i] > 2.0: print(factor, e)

            #print(e_gen)
            #print(e_reco)


            factor_theta = np.random.normal(loc=1.0, scale=0.001, size=nTrks_gen)
            factor_phi = np.random.normal(loc=1.0, scale=0.001, size=nTrks_gen)

            theta_reco = theta_gen * factor_theta
            phi_reco = phi_gen * factor_phi
            #theta_reco = theta_gen
            #phi_reco = phi_gen

            m_reco = m_gen

            p_reco = np.sqrt(e_reco**2-m_reco**2)

            px_reco = p_reco * np.sin(theta_reco) * np.cos(phi_reco)
            py_reco = p_reco * np.sin(theta_reco) * np.sin(phi_reco)
            pz_reco = p_reco * np.cos(theta_reco)

            pt_reco = p_reco * np.sin(theta_reco)

            sel = pt_reco > 0.2

            px_reco = px_reco[sel]
            py_reco = py_reco[sel]
            pz_reco = pz_reco[sel]
            p_reco = p_reco[sel]
            m_reco = m_reco[sel]
            theta_reco = theta_reco[sel]
            phi_reco = phi_reco[sel]

            nTrks_reco = len(px_reco)
            if nTrks_reco == 0: continue
            if np.sum(e_reco) > 200: continue
            sphericity_reco = calculate_sphericity(px_reco, py_reco, pz_reco)
            if abs(sphericity_reco["cos_theta_v1"])>0.82: continue

            self._hists['counter'].Fill(1.5)

            dists = np.full((nTrks_reco, nTrks_gen), -1, 'd')

            # matching
            for i in range(nTrks_reco):
                for j in range(nTrks_gen):
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
            func = ROOT.TF1(f"smr_e{i}", gaussian, 0.9, 1.1, 2)
            func.SetParameters(*params)
            self._pdfs.append(func)

    def loadSMRParamsCB2(self, fparams):
        with open(fparams, "r") as file:
            params = json.load(file)
             
        for i, param in enumerate(params):
            func = ROOT.TF1(f"smr_e{i}", double_sided_crystal_ball, 0.9, 1.1, 7)
            func.SetParameters(*param['params'])
            #print(param['params'])
            self._pdfs.append(func)

if __name__ == "__main__":


    filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/LEP1MC1994_recons_aftercut-032.root'
    filenameout = "smeared_response.root"

    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    s=int(time.time())
    ROOT.gRandom.SetSeed(s)
    
    tgen = 'tgen'

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
    #response.loadSMRParams("fit_results.json")
    response.loadSMRParamsCB2("fit_results_cb2.json")
    response.bookHistograms()
    response.bookResponseMatrices()
    response.loop()
    #response.normalize()
    response.writeToFile(fnameout)
