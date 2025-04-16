import ROOT
import math
import numpy as np
import sys
import time
import argparse
from itertools import combinations_with_replacement, permutations
from array import array

class E3CAnalyzer:
    
    def __init__(self, reco_tree, treename, outname):
        self._hists = {}
        self._treco = reco_tree
        self._fout = ROOT.TFile(outname, 'RECREATE')
        hname = 'counter'
        self._evt_counter = ROOT.TH1D(hname, hname, 2, 0, 2)
        self._otree1 = ROOT.TTree("eec", "flat output tree for eec plotting")
        self._otree2 = ROOT.TTree("e3c", "flat output tree for e3c plotting")
        self._otree3 = ROOT.TTree("e3c_full", "flat output tree for collinear safe e3c")
        self._otree4 = ROOT.TTree("e3c_spin", "flat output tree for spin correlation")
        self._isALEPHGen = (treename == 'tgen')
        self._isGen = ("SHERPA" in outname or "HERWIG" in outname or "Belle" in outname) 
        self._isPythia = ("PYTHIA" in outname)
        self._useJet = False
        self._studySpin = False

    def addJetTree(self, jet_tree):
        self._useJet = True
        self._tjet = jet_tree

    def calcBinEdgeDoubleLog(self, low, high, nbins):   
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

        return np.array(bin_edge)

    def calcBinEdgeLog(self, low, high, nbins):   
        xlow = np.log10(low)
        xhigh = np.log10(high)
        width = (xhigh-xlow)/nbins

        bins=[]
        for i in range(nbins+1):
            val = pow(10, xlow + i * width)
            bins += [val]
        return np.array(bins)

    def calcNormV(self, A, B):
        norm_vector = np.cross(A, B)
        return norm_vector

    def calcAngle(self, n1, n2):
        cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  

        theta = np.arccos(cos_theta)
        return theta

    def calcAngleWithDirection(self, n1, n2):
        cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  
        theta = np.arccos(cos_theta)
        cross = np.cross(n1, n2)
        if cross[0] > 0: return theta
        else: return 2*np.pi-theta

    def initializeTrees(self):
        tree_vars1 = {
            '_eec': 'eec', '_r': 'r', '_z': 'z'
        }
        for var, name in tree_vars1.items():
            setattr(self, var, array('f', [0]))
            self._otree1.Branch(name, getattr(self, var), f"{name}/F")

        tree_vars2 = {
            '_e3c': 'e3c', '_rL': 'rL', '_rM': 'rM', '_rS': 'rS', 
            '_x': 'x', '_y': 'y', '_zeta': 'zeta', '_phi': 'phi'
        }

        for var, name in tree_vars2.items():
            setattr(self, var, array('f', [0]))
            self._otree2.Branch(name, getattr(self, var), f"{name}/F")

        tree_vars3 = {
            '_e3c_full': 'e3c_full', '_r_full': 'r_full'
        }
        for var, name in tree_vars3.items():
            setattr(self, var, array('f', [0]))
            self._otree3.Branch(name, getattr(self, var), f"{name}/F")

    def initializeSpinTrees(self):
        self._studySpin = True
        spin_vars = {'_e3cS': 'e3cS', '_phiS': 'phiS', '_rLS': 'rLS', '_rSS': 'rSS'}
        for var, name in spin_vars.items():
            setattr(self, var, array('f', [0]))
            self._otree4.Branch(name, getattr(self, var), f"{name}/F")

    def bookHistograms(self):
        doubleLog1 = self.calcBinEdgeDoubleLog(0.001, np.pi/2, 100)
        doubleLog2 = self.calcBinEdgeDoubleLog(0.001, np.pi/2, 150)
        log = self.calcBinEdgeLog(0.001, 1, 200)

        phibins = 60
        phibinning = np.linspace(0, 2*np.pi, phibins+1)
        log2 = self.calcBinEdgeLog(0.0001, 1, 400)

        hist_params = {
            "cpt": (1000, 0, 100),
            "ceta": (100, -5, 5),
            "cphi": (80, -4, 4),
            "eec_r": (len(doubleLog1)-1, doubleLog1),
            "e3c_rL": (len(doubleLog2)-1, doubleLog2),
            "e3c_rM": (len(doubleLog2)-1, doubleLog2),
            "e3c_rM_linear": (200, 0, np.pi),
            "e3c_rS": (len(doubleLog2)-1, doubleLog2),
            "e3c_rS_linear": (200, 0, 2.1),
            "e3c_zeta": (len(log)-1, log),
            "e3c_phi": (200, 0, np.pi/2),
            "e3c_x": (200, 0.5, 1),
            "e3c_y": (200, 0, 1),
            "e3c_xy": (200, 0.5, 1, 200, 0, 1),
            "e3c_r": (len(doubleLog2)-1, doubleLog2),
            "spin_rL": (len(doubleLog1)-1, doubleLog1),
            "spin_rS": (len(doubleLog1)-1, doubleLog1),
            "spin_phi": (phibins, phibinning),
            "spin_2d": (phibins, phibinning, len(log2)-1, log2)
        }
        for name, params in hist_params.items():
            if len(params) <= 3:
                self._hists[name] = ROOT.TH1D(name, "", *params)
            else:
                self._hists[name] = ROOT.TH2D(name, "", *params)

        if self._useJet:
            self._hists['e3c_rM_multij'] = ROOT.TH1D('e3c_rM_multij', '', len(doubleLog2)-1, doubleLog2)


    def normalizeByBinWidth(self, h):
        for b in range(h.GetNbinsX()):
            h.SetBinContent(b+1, h.GetBinContent(b+1)/h.GetBinWidth(b+1))
            h.SetBinError(b+1, h.GetBinError(b+1)/h.GetBinWidth(b+1))
        return h

    def find_unique_and_repetitive(self, arr):
        seen = set()
        repetitive = set()
    
        for num in arr:
            if num in seen:
                repetitive.add(num)
            else:
                seen.add(num)
    
        unique = seen - repetitive 
        return list(unique), list(repetitive)

    def loopAndFillTrees(self):
        nEntries = self._treco.GetEntries()
        nEntries = 40000 ### no more than 40000 events to reduce the disk space usage !!!
        #nEntries = 1
        for iEvt in range(nEntries):

            if iEvt % 1000 == 0:
                print(f"Processed {iEvt} events")

            self._treco.GetEntry(iEvt)

            self._evt_counter.Fill(0.5)
    
            if not self._isGen and not self._isALEPHGen and (self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5): continue
            if self._isALEPHGen and self._treco.passesSTheta < 0.5: continue
                
            self._evt_counter.Fill(1.5)
    
            E = self._treco.Energy
            
            px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco, hp_reco = (
                np.array(getattr(self._treco, attr)) for attr in ['px', 'py', 'pz', 'mass', 'charge', 'theta', 'pt', 'eta', 'phi', 'highPurity']
            )
            sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
            px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco = (
                arr[sel_reco] for arr in [px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco]
            )
    
            Ei_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2 + m_reco**2)
            p3 = np.stack((px_reco, py_reco, pz_reco), axis=1)
            ntrk = len(p3)
            
            dot_products = p3 @ p3.T
            p3_mag = np.linalg.norm(p3, axis=1)
            outer_mag = np.outer(p3_mag, p3_mag)
            
            cos_similarity = np.clip(dot_products / outer_mag, -1.0, 1.0)
            distances = np.arccos(cos_similarity)
    
            indices = np.arange(ntrk)    
            for p in combinations_with_replacement(indices, 3):
                if len(set(p)) == 1: continue

                i, j, k = p
                if len(set(p)) == 3:
                    R = sorted([(distances[i][j], (i, j)),
                                    (distances[i][k], (i, k)),
                                    (distances[j][k], (j, k))], key=lambda x: x[0])
                    E3 = Ei_reco[i]*Ei_reco[j]*Ei_reco[k]/E**3
                    if not self._studySpin:
                        
                        self._e3c[0] = E3
                        self._e3c_full[0] = 6*E3
                        
                        self._rS[0] = R[0][0]
                        self._rM[0] = R[1][0]
                        self._rL[0] = R[2][0]
                        self._r_full[0] = self._rL[0]
                        self._zeta[0] = self._rS[0]/self._rM[0]
                        
                        if self._rS[0] == 0: self._rS[0] = rS = 1e-6
                        ratio = (self._rL[0] - self._rM[0])/self._rS[0]
                        val = max(0, 1 - ratio**2)
                        self._phi[0] = math.asin(math.sqrt(val))
                        
                        cos = (self._rM[0]**2+self._rL[0]**2-self._rS[0]**2)/(2*self._rM[0]*self._rL[0])
                        cos = min(1, max(-1, cos))
                        val = max(0, 1 - cos**2)
                        self._x[0] = cos*self._rM[0]/self._rL[0]
                        self._y[0] = math.sqrt(val)*self._rM[0]/self._rL[0]
    
                        self._otree2.Fill()
                        self._otree3.Fill()

                    else:
                        long_edge = R[2][1]
                        short_edge = R[0][1]
                        common_vertex = set(long_edge).intersection(short_edge).pop()

                        v1 = long_edge[0] if long_edge[1] == common_vertex else long_edge[1]
                        v2, v3 = short_edge
                        m1 = p3[v1]
                        m2 = p3[v2]
                        m3 = p3[v3]

                        m2m3 = m2 + m3
                            
                        n1 = self.calcNormV(m1, m2m3)
                        n2 = self.calcNormV(m2, m3)
                            
                        self._e3cS[0] = E3
                        self._phiS[0] = self.calcAngleWithDirection(n1, n2)
                        self._rLS[0] = self.calcAngle(m1, m2m3)
                        self._rSS[0] = self.calcAngle(m2, m3)
                            
                        self._otree4.Fill()
                    
                else:
                    if not self._studySpin:
                        
                        unique, repetitive = self.find_unique_and_repetitive(p)
                        m = repetitive[0]
                        n = unique[0]
                        self._eec[0] = 2*Ei_reco[m]*Ei_reco[n]/E**2
                        self._r[0] = distances[m][n]
                        self._z[0] = 0.5*(1-np.cos(distances[m][n]))
                        self._e3c_full[0] = 3*(Ei_reco[m]**2)*Ei_reco[n]/E**3
                        self._r_full[0] = distances[m][n]
                        
                        self._otree1.Fill()
                        self._otree3.Fill()

    def loopAndFillHists(self):
        nEntries = self._treco.GetEntries()
        #nEntries = 100
        for iEvt in range(nEntries):

            if iEvt % 10000 == 0:
                print(f"Processed {iEvt} events")
            
            self._treco.GetEntry(iEvt)

            if self._useJet: self._tjet.GetEntry(iEvt)

            self._evt_counter.Fill(0.5)
    
            if not self._isGen and not self._isALEPHGen and (self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5): continue
            if self._isALEPHGen and not self._isPythia and self._treco.passesSTheta < 0.5: continue
            if self._isPythia and self._treco.passSphericity < 0.5: continue
                
            self._evt_counter.Fill(1.5)

            try:
                E = self._treco.Energy
            except:
                E = 90

            if self._isPythia:
                px_reco, py_reco, pz_reco, m_reco, c_reco, mom, pt_reco, eta_reco, phi_reco = (
                np.array(getattr(self._treco, attr)) for attr in ['px', 'py', 'pz', 'mass', 'isCharged', 'p', 'pt', 'eta', 'phi']
                )
                theta_reco = np.arcsin(pt_reco/mom)
                sel_reco = (abs(c_reco) > 0.5) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348)
            else:
                px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco, hp_reco = (
                    np.array(getattr(self._treco, attr)) for attr in ['px', 'py', 'pz', 'mass', 'charge', 'theta', 'pt', 'eta', 'phi', 'highPurity']
                )
                sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
                
            #print(sel_reco, px_reco)
            px_reco, py_reco, pz_reco, m_reco, theta_reco, pt_reco, eta_reco, phi_reco = (
                arr[sel_reco] for arr in [px_reco, py_reco, pz_reco, m_reco, theta_reco, pt_reco, eta_reco, phi_reco]
            )
    
            Ei_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2 + m_reco**2)
            p3 = np.stack((px_reco, py_reco, pz_reco), axis=1)
            ntrk = len(p3)
            
            dot_products = p3 @ p3.T
            p3_mag = np.linalg.norm(p3, axis=1)
            outer_mag = np.outer(p3_mag, p3_mag)
            
            cos_similarity = np.clip(dot_products / outer_mag, -1.0, 1.0)
            distances = np.arccos(cos_similarity)
    
            indices = np.arange(ntrk)
            pairs = list(combinations_with_replacement(indices, 3))
            for p in sorted(pairs):
                if len(set(p)) == 1:
                    t = p[0]
                    self._hists['cpt'].Fill(pt_reco[t])
                    self._hists['ceta'].Fill(eta_reco[t])
                    self._hists['cphi'].Fill(phi_reco[t])
                if len(set(p)) == 3:
                    i = p[0]
                    j = p[1]
                    k = p[2]
                    E3 = Ei_reco[i]*Ei_reco[j]*Ei_reco[k]/E**3
                    e3c = E3
                    e3c_full = 6*E3
                    R = sorted([(distances[i][j], (i, j)),
                                (distances[i][k], (i, k)),
                                (distances[j][k], (j, k))], key=lambda x: x[0])
                    rS = R[0][0]
                    rM = R[1][0]
                    rL = R[2][0]
                    r_full = R[2][0]
                    zeta = rS/rM
                    if rS == 0: rS = 0.0000001
                    ratio = (rL - rM)/rS
                    val = max(0, 1 - ratio**2)
                    phi = math.asin(math.sqrt(val))
                    cos = (rM**2+rL**2-rS**2)/(2*rM*rL)
                    cos = min(1, max(-1, cos))
                    val = max(0, 1 - cos**2)
                    x = cos*rM/rL
                    y = math.sqrt(val)*rM/rL

                    long_edge = R[2][1]
                    short_edge = R[0][1]
                    common_vertex = set(long_edge).intersection(short_edge).pop()

                    v1 = long_edge[0] if long_edge[1] == common_vertex else long_edge[1]
                    v2, v3 = short_edge
                    m1 = p3[v1]
                    m2 = p3[v2]
                    m3 = p3[v3]

                    m2m3 = m2 + m3
                            
                    n1 = self.calcNormV(m1, m2m3)
                    n2 = self.calcNormV(m2, m3)
                    spin_phi = self.calcAngleWithDirection(n1, n2)
                    spin_rL = self.calcAngle(m1, m2m3)
                    spin_rS = self.calcAngle(m2, m3)
                    
                    self._hists['e3c_rL'].Fill(rL, e3c)
                    self._hists['e3c_rM'].Fill(rM, e3c)
                    self._hists['e3c_rS'].Fill(rS, e3c)
                    self._hists['e3c_rM_linear'].Fill(rM, e3c)
                    if self._useJet:
                        if njet[0] >=3: self._hists['e3c_rM_multij'].Fill(rM, e3c)
                    self._hists['e3c_rS_linear'].Fill(rS, e3c)
                    self._hists['e3c_r'].Fill(rL, e3c_full)
                    self._hists['e3c_r'].Fill(rM, e3c_full)
                    self._hists['e3c_r'].Fill(rS, e3c_full)
                    
                    self._hists['e3c_x'].Fill(x, e3c)
                    self._hists['e3c_y'].Fill(y, e3c)
                    self._hists['e3c_xy'].Fill(x, y, e3c)
                    self._hists['e3c_zeta'].Fill(zeta, e3c)
                    self._hists['e3c_phi'].Fill(phi, e3c)

                    self._hists['spin_rL'].Fill(spin_rL, e3c)
                    self._hists['spin_rS'].Fill(spin_rS, e3c)
                    self._hists['spin_2d'].Fill(spin_phi, spin_rS**2, e3c*spin_rL**2*spin_rS**2)
                    if spin_rL**2 < 1 and spin_rS**2 < 0.1:
                        self._hists['spin_phi'].Fill(spin_phi, e3c*spin_rL**2*spin_rS**2)
                    
                if len(set(p)) == 2:
                    unique, repetitive = self.find_unique_and_repetitive(p)
                    m = repetitive[0]
                    n = unique[0]
                    eec = 2*Ei_reco[m]*Ei_reco[n]/E**2
                    r = distances[m][n]
                    e3c_full = 3*(Ei_reco[m]**2)*Ei_reco[n]/E**3
                    self._hists['e3c_r'].Fill(r, e3c_full)
                    self._hists['eec_r'].Fill(r, eec)  

    def writeTrees(self):
        self._fout.cd()
        self._evt_counter.Write()
        self._otree1.Write()
        self._otree2.Write()
        self._otree3.Write()
        self._fout.Close()

    def writeSpinTrees(self):
        self._fout.cd()
        self._evt_counter.Write()
        self._otree4.Write()
        self._fout.Close()

    def writeHistograms(self):
        self._fout.cd()
        self._evt_counter.Write()
        for h in self._hists.keys():
            self._hists[h].Write()
        self._fout.Close()
        

if __name__ == "__main__":

    filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPH/LEP1Data1994P1_recons_aftercut-MERGED.root'
    #filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/LEP1MC1994_recons_aftercut-001.root'
    #filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/HERWIG715/run_00/Belle_0_0.root'
    filenameout = 'h_'+filename.split('/')[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    treename = 'tgen'
    fin = ROOT.TFile.Open(args.infiles, 'r')
    t_hadrons = fin.Get(treename)

    t_jet = fin.Get('akR4ESchemeJetTree')

    start = time.perf_counter()

    analyzer = E3CAnalyzer(t_hadrons, treename, args.outfile)
    #analyzer.addJetTree(t_jet)
    analyzer.bookHistograms()
    analyzer.loopAndFillHists()
    analyzer.writeHistograms()
    
    #analyzer.initializeTrees()
    #analyzer.loopAndFillTrees()
    #analyzer.writeTrees()

    #analyzer.initializeSpinTrees()
    #analyzer.loopAndFillTrees()
    #analyzer.writeSpinTrees()

    end = time.perf_counter()

    elapsed = int(end - start)
    minutes = elapsed // 60
    seconds = elapsed % 60

    print(f"Total elapsed time: {minutes} min {seconds} sec")
    
