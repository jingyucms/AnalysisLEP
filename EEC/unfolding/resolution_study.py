import ROOT
import numpy as np
import sys
import os
import argparse
import math
from scipy.optimize import linear_sum_assignment
from array import array

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

doAngular = True
if doAngular:
    matching_r = 0.05
else:
    matching_r = 10000

class MyResolution:

    def __init__(self, reco_tree, gen_tree, outname):
        self._hists = {}
        self._resps = {}
        self._treco = reco_tree
        self._tgen = gen_tree
        self._evt_counter = 0

        self._fout = ROOT.TFile(outname, 'RECREATE')
        self._otree = ROOT.TTree("resolution", "resolution")

        self._evt_counter = ROOT.TH1F("counter", "counter", 2, 0, 2)
        

    def writeToFile(self):
        self._fout.cd()
        self._evt_counter.Write()
        self._otree.Write()
        self._fout.Close()

    def init_tree(self):
        tree_vars = {
            '_respE': 'respE', '_respT': 'respT', '_respF': 'respF',
            '_genE': 'genE', '_genT': 'genT', '_genF': 'genF'
        }
        for var, name in tree_vars.items():
            setattr(self, var, array('f', [0]))
            self._otree.Branch(name, getattr(self, var), f"{name}/F")
        
    def loop(self):

        nevt = self._treco.GetEntries()
        for n in range(nevt):

            self._evt_counter.Fill(0.5)

            self._tgen.GetEntry(n)
            self._treco.GetEntry(n)
            
            ## gen event selection
            if self._tgen.passesSTheta < 0.5: continue

            ## reco event selection
            if self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5: continue

            self._evt_counter.Fill(1.5)

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
            
            nTrks_reco = len(px_reco)
            nTrks_gen = len(px_gen)
            
            #if np.sum(e_gen) > 200: continue

            p3_gen = np.stack((px_gen, py_gen, pz_gen), axis=1)
            p3_reco = np.stack((px_reco, py_reco, pz_reco), axis=1)

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
            for pair in matched:
                #print(matched)
                v1, v2 = int(pair[0]), int(pair[1])
                
                pxi = px_reco[v1]
                pyi = py_reco[v1]
                pzi = pz_reco[v1]
                mi = m_reco[v1]
                itheta = theta_reco[v1]
                iphi = phi_reco[v1]
        
                pxj = px_gen[v2]
                pyj = py_gen[v2]
                pzj = pz_gen[v2]
                mj = m_gen[v2]
                jtheta = theta_gen[v2]
                jphi = phi_gen[v2]

                ei = E(pxi, pyi, pzi, mi)
                ej = E(pxj, pyj, pzj, mj)
                
                self._respE[0] = pt_reco[v1] / pt_gen[v2]
                self._genE[0] = pt_gen[v2]

                self._respT[0] = itheta / jtheta
                self._genT[0] = jtheta

                self._respF[0] = 0 if jphi == 0 else iphi / jphi
                self._genF[0] = jphi

                self._otree.Fill()
                
                
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
    filenameout = "resolution.root"

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

    response = MyResolution(t_reco, t_gen, fnameout)
    response.init_tree()
    response.loop()
    response.writeToFile()
