import ROOT
import math
import numpy as np
from array import array
import sys
import time
import argparse
from itertools import combinations_with_replacement, permutations

class E3CAnalyzer:
    
    def __init__(self, reco_tree, treename, outname):
        self._hists = {}
        self._treco = reco_tree
        self._fout = ROOT.TFile(outname, 'RECREATE')
        hname = 'counter'
        self._evt_counter = ROOT.TH1F(hname, hname, 2, 0, 2)
        self._otree1 = ROOT.TTree("eec", "flat output tree for eec plotting")
        self._otree2 = ROOT.TTree("e3c", "flat output tree for e3c plotting")
        self._otree3 = ROOT.TTree("e3c_full", "flat output tree for collinear safe e3c")
        self._otree4 = ROOT.TTree("e3c_spin", "flat output tree for spin correlation")
        self._isGen = (treename == 'tgen')
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

        theta = np.arccos(np.abs(cos_theta))
        return theta
        
    def initializeTrees(self):
        self._eec = array('f',[0])
        self._r = array('f',[0])
        self._z = array('f',[0])
    
        self._e3c = array('f',[0])
        self._rL = array('f',[0])
        self._rM = array('f',[0])
        self._rS = array('f',[0])
        self._x = array('f',[0])
        self._y = array('f',[0])
        self._zeta = array('f',[0])
        self._phi = array('f',[0])

        self._e3c_full = array('f',[0])
        self._r_full = array('f',[0])
    
        self._otree1.Branch("eec", self._eec, "eec/F")
        self._otree1.Branch("r", self._r, "r/F")
        self._otree1.Branch("z", self._z, "z/F")
        
        self._otree2.Branch("e3c", self._e3c, "e3c/F")
        self._otree2.Branch("rL", self._rL, "rL/F")
        self._otree2.Branch("rM",self._rM, "rM/F")
        self._otree2.Branch("rS", self._rS, "rS/F")
        self._otree2.Branch("x", self._x, "x/F")
        self._otree2.Branch("y", self._y, "y/F")
        self._otree2.Branch("zeta", self._zeta, "zeta/F")
        self._otree2.Branch("phi", self._phi, "phi/F")

        self._otree3.Branch("e3c_full", self._e3c_full, "e3c_full/F")
        self._otree3.Branch("r_full", self._r_full, "r_full/F")

    def initializeSpinTrees(self):
        self._studySpin = True

        self._e3cS = array('f',[0])
        self._phiS = array('f',[0])
        self._rLS = array('f',[0])
        self._rSS = array('f',[0])

        self._otree4.Branch("e3cS", self._e3cS, "e3cS/F")
        self._otree4.Branch("phiS", self._phiS, "phiS/F")
        self._otree4.Branch("rLS", self._rLS, "rLS/F")
        self._otree4.Branch("rSS", self._rSS, "rSS/F")

    def bookHistograms(self):

        doubleLog1 = self.calcBinEdgeDoubleLog(0.001, np.pi/2, 100)
        doubleLog2 = self.calcBinEdgeDoubleLog(0.001, np.pi/2, 150)
        log = self.calcBinEdgeLog(0.001, 1, 200)

        hname = "cpt"
        self._hists[hname] = ROOT.TH1F(hname, "", 1000, 0, 100)

        hname = "ceta"
        self._hists[hname] = ROOT.TH1F(hname, "", 100, -5, 5)

        hname = "cphi"
        self._hists[hname] = ROOT.TH1F(hname, "", 80, -4 ,4)

        hname = 'eec_r'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog1)-1, doubleLog1)

        hname = 'e3c_rL'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog2)-1, doubleLog2)

        hname = 'e3c_rM'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog2)-1, doubleLog2)

        hname = 'e3c_rM_linear'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0, np.pi)

        if self._useJet:
            hname = 'e3c_rM_multij'
            self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog2)-1, doubleLog2)

        hname = 'e3c_rS'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog2)-1, doubleLog2)

        hname = 'e3c_rS_linear'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0, 2.1)

        hname = 'e3c_zeta'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(log)-1, log)

        hname = 'e3c_phi'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0, np.pi/2)

        hname = 'e3c_x'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0.5, 1)

        hname = 'e3c_y'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0, 1)

        hname = 'e3c_phiS'
        self._hists[hname] = ROOT.TH1F(hname, hname, 200, 0, np.pi)

        hname = 'e3c_xy'
        self._hists[hname] = ROOT.TH2F(hname, hname, 200, 0.5, 1, 200, 0, 1)

        hname = 'e3c_r'
        self._hists[hname] = ROOT.TH1F(hname, hname, len(doubleLog2)-1, doubleLog2)

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
        nEntries = 10000 ### no more than 10000 events to reduce the disk space usage !!!
            self._treco.GetEntry(iEvt)

            self._evt_counter.Fill(0.5)
    
            if self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5: continue
                
            self._evt_counter.Fill(1.5)
    
            E = self._treco.Energy
            
            px_reco = np.array(self._treco.px)
            py_reco = np.array(self._treco.py)
            pz_reco = np.array(self._treco.pz)
            m_reco = np.array(self._treco.mass)
            c_reco = np.array(self._treco.charge)
            theta_reco = np.array(self._treco.theta)
            pt_reco = np.array(self._treco.pt)
            hp_reco = np.array(self._treco.highPurity)
    
            sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
    
            px_reco = px_reco[sel_reco]
            py_reco = py_reco[sel_reco]
            pz_reco = pz_reco[sel_reco]
            m_reco = m_reco[sel_reco]
            c_reco = c_reco[sel_reco]
            theta_reco = theta_reco[sel_reco]
            pt_reco = pt_reco[sel_reco]
    
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
                    if not self._studySpin:
                        E3 = Ei_reco[i]*Ei_reco[j]*Ei_reco[k]/E**3
                        self._e3c[0] = E3
                        self._e3c_full[0] = 6*E3
                        
                        R = sorted([(distances[i][j], (i, j)),
                                    (distances[i][k], (i, k)),
                                    (distances[j][k], (j, k))], key=lambda x: x[0])
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
                        for perm in permutations(p):
                            v1, v2, v3 = perm
                            if v2 > v3: continue
                            m1 = p3[v1]
                            m2 = p3[v2]
                            m3 = p3[v3]

                            m2m3 = m2 + m3
                            
                            n1 = self.calcNormV(m1, m2m3)
                            n2 = self.calcNormV(m2, m3)

                            E3 = Ei_reco[v1]*Ei_reco[v2]*Ei_reco[v3]/E**3
                            
                            self._e3cS[0] = E3
                            self._phiS[0] = self.calcAngle(n1, n2)
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
            self._treco.GetEntry(iEvt)

            if self._useJet: self._tjet.GetEntry(iEvt)

            self._evt_counter.Fill(0.5)
    
            if not self._isGen and (self._treco.passesSTheta < 0.5 or self._treco.passesNTrkMin < 0.5 or self._treco.passesTotalChgEnergyMin < 0.5): continue
            if self._isGen and self._treco.passesSTheta < 0.5: continue
                
            self._evt_counter.Fill(1.5)
    
            E = self._treco.Energy
            nref = self._tjet.nref
            njet[0] = self._tjet.jtN if self._useJet else 0
            
            px_reco = np.array(self._treco.px)
            py_reco = np.array(self._treco.py)
            pz_reco = np.array(self._treco.pz)
            m_reco = np.array(self._treco.mass)
            c_reco = np.array(self._treco.charge)
            theta_reco = np.array(self._treco.theta)
            pt_reco = np.array(self._treco.pt)
            eta_reco = np.array(self._treco.eta)
            phi_reco = np.array(self._treco.phi)
            hp_reco = np.array(self._treco.highPurity)
    
            sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
    
            px_reco = px_reco[sel_reco]
            py_reco = py_reco[sel_reco]
            pz_reco = pz_reco[sel_reco]
            m_reco = m_reco[sel_reco]
            c_reco = c_reco[sel_reco]
            theta_reco = theta_reco[sel_reco]
            pt_reco = pt_reco[sel_reco]
            eta_reco = eta_reco[sel_reco]
            phi_reco = phi_reco[sel_reco]
    
            Ei_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2 + m_reco**2)
    
            ntrk = len(px_reco)
    
            p3 = np.stack((px_reco, py_reco, pz_reco), axis=1)
            dot_products = np.dot(p3, p3.T)
            p3_mag = np.linalg.norm(p3, axis=1)
    
            p3_mag = np.round(p3_mag, 8)
            dot_products = np.round(dot_products, 8)
            
            cos_similarity = np.round(dot_products / np.outer(p3_mag, p3_mag), 8)
    
            cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
            
            distances = np.round(np.arccos(cos_similarity), 8)
    
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

                    short_edge = R[0][1]
                    long_edge = R[2][1]
                    
                    common_vertex = set(long_edge).intersection(short_edge).pop()

                    v1, v2 = long_edge
                    v3 = short_edge[1] if short_edge[0] == common_vertex else short_edge[0]
                    vertices = [v1, v2, v3]

                    p1 = np.array([px_reco[v1], py_reco[v1], pz_reco[v1]])
                    p2 = np.array([px_reco[v2], py_reco[v2], pz_reco[v2]])
                    p3 = np.array([px_reco[v3], py_reco[v3], pz_reco[v3]])

                    n1 = self.calcNormV(p1, p2+p3)
                    n2 = self.calcNormV(p2, p3)

                    phiS = self.calcAngle(n1, n2)
                    
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
                    self._hists['e3c_phiS'].Fill(phiS, e3c)
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
    filenameout = 't_'+filename.split('/')[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    treename = 't'
    fin = ROOT.TFile.Open(args.infiles, 'r')
    t_hadrons = fin.Get(treename)

    t_jet = fin.Get('akR4ESchemeJetTree')

    start = time.perf_counter()

    analyzer = E3CAnalyzer(t_hadrons, treename, args.outfile)
    #analyzer.addJetTree(t_jet)
    #analyzer.bookHistograms()
    #analyzer.loopAndFillHists()
    #analyzer.writeHistograms()
    
    #analyzer.initializeTrees()
    #analyzer.loopAndFillTrees()
    #analyzer.writeTrees()

    analyzer.initializeSpinTrees()
    analyzer.loopAndFillTrees()
    analyzer.writeSpinTrees()

    end = time.perf_counter()

    elapsed = int(end - start)
    minutes = elapsed // 60
    seconds = elapsed % 60

    print(f"Total elapsed time: {minutes} min {seconds} sec")
    
