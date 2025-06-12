import ROOT
import math
import numpy as np
from array import array
import itertools
import argparse
import sys

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

def calcAngle(n1, n2):
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    
    theta = np.arccos(cos_theta)
    return theta

def conversion_veto_mask(eta, phi, charge, pwflag,
                         conversion_deta, conversion_dphi):
    """
    Parameters
    ----------
    eta, phi, charge, pwflag : array-like, shape (N,)
        Per-track arrays.
        - pwflag==2 marks an electron.
    conversion_deta : float
        Maximum |Δη| between candidate and its partner.
    conversion_dphi : float
        Maximum |Δφ| between candidate and its partner.
    
    Returns
    -------
    mask : ndarray of bool, shape (N,)
        True for tracks that are NOT conversion electrons,
        False for conversions.
    """
    eta   = np.asarray(eta)
    phi   = np.asarray(phi)
    charge= np.asarray(charge)
    pw    = np.asarray(pwflag)
    N     = eta.size

    # Build an array of “previous” indices,
    # with index 0 peeking at 1 so we don’t run off the front:
    prev = np.arange(N) - 1
    prev[0] = 1

    # 1) both must be electrons (pwflag==2)
    both_elec = (pw == 2) & (pw[prev] == 2)
    # 2) opposite charge
    opp_charge = (charge == -charge[prev])
    # 3) |Δη| small
    deta = np.abs(eta - eta[prev])
    pass_deta  = (deta <= conversion_deta)
    # 4) |Δφ| small, accounting for periodicity
    dphi = np.arccos(np.cos(phi - phi[prev]))
    pass_dphi  = (dphi <= conversion_dphi)

    # conversion if all four conditions hold
    is_conv = both_elec & opp_charge & pass_deta & pass_dphi

    # now propagate back onto prev[i]
    conv_prev = np.zeros_like(is_conv, dtype=bool)
    conv_prev[prev] = is_conv

    # any track that is_conv or conv_prev is a conversion
    conv_any = is_conv | conv_prev

    # final mask: True = keep (not conversion), False = drop
    return ~conv_any


eijbins2 = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 
0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

bins_theta = calcBinEdge(0.002, np.pi/2, 100)
bins_z = calcBinEdge(0.000001, 0.5, 100)

# 0) A one‐bin “counter” histogram
h0 = ROOT.TH1D("N", "", 2, 0, 2)

# 1) Define all your 1D histos and their bin edges
h1d_defs = {
    "EEC_r"       : bins_theta,
    "EEC_z"       : bins_z,
    "EEC_r_pos"   : bins_theta,
    "EEC_z_pos"   : bins_z,
    "EEC_r_neg"   : bins_theta,
    "EEC_z_neg"   : bins_z,
    "EEC_r_cross" : bins_theta,
    "EEC_z_cross" : bins_z,
    "eta_conv_ele": np.linspace(0, 4, 41), 
}

# Build them in a single comprehension
h1d = {
    name: ROOT.TH1D(name, "", len(edges)-1, np.array(edges))
    for name, edges in h1d_defs.items()
}

# 2) Define your 2D histos: x–bins come from bins_theta/z, y–bins from eijbins2
h2d_defs = {
    "EEC2d_r"       : (bins_theta, eijbins2),
    "EEC2d_z"       : (bins_z,     eijbins2),
    "EEC2d_r_pos"   : (bins_theta, eijbins2),
    "EEC2d_z_pos"   : (bins_z,     eijbins2),
    "EEC2d_r_neg"   : (bins_theta, eijbins2),
    "EEC2d_z_neg"   : (bins_z,     eijbins2),
    "EEC2d_r_cross" : (bins_theta, eijbins2),
    "EEC2d_z_cross" : (bins_z,     eijbins2),
}

h2d = {
    name: ROOT.TH2D(
        name, "",
        len(xedges)-1, np.array(xedges),
        len(yedges)-1, np.array(yedges)
    )
    for name, (xedges, yedges) in h2d_defs.items()
}

HISTS = {
    'all':    (h1d['EEC_r'], h1d['EEC_z'],
               h2d['EEC2d_r'], h2d['EEC2d_z']),
    'pos':    (h1d['EEC_r_pos'], h1d['EEC_z_pos'],
               h2d['EEC2d_r_pos'], h2d['EEC2d_z_pos']),
    'neg':    (h1d['EEC_r_neg'], h1d['EEC_z_neg'],
               h2d['EEC2d_r_neg'], h2d['EEC2d_z_neg']),
    'cross':  (h1d['EEC_r_cross'], h1d['EEC_z_cross'],
               h2d['EEC2d_r_cross'], h2d['EEC2d_z_cross']),
}

def fill_pair(i, j, r, z, eec, c_reco):
    # determine category
    cat = 'neg' if c_reco[i]<0 and c_reco[j]<0 else \
          'pos' if c_reco[i]>0 and c_reco[j]>0 else 'cross'
    for tag in ('all', cat):
        h1, h2, h3, h4 = HISTS[tag]
        h1.Fill(r,   eec)
        h2.Fill(z,   eec)
        h3.Fill(r,   eec)
        h4.Fill(z,   eec)

if __name__ == "__main__":

    filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPH/LEP1Data1994P1_recons_aftercut-MERGED.root'
    #filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/ALEPHMC/LEP1MC1994_recons_aftercut-001.root'
    #filename = '/eos/user/z/zhangj/ALEPH/SamplesLEP1/HERWIG715/run_00/Belle_0_0.root'
    filenameout = 'h_'+filename.split('/')[-1]
    isGen = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='?', default=filename, help="name of input files")
    parser.add_argument("outfile", nargs='?', default=filenameout, help="name of input files")
    args = parser.parse_args()

    treename = 't'
    if isGen: treename = 'tt'
    
    fin = ROOT.TFile.Open(args.infiles, 'r')
    t_hadrons = fin.Get(treename)

    fout = ROOT.TFile(args.outfile, 'RECREATE')

    ot = ROOT.TTree("eec","flat output")
    arrs = {n: array(ctypes, [0]) for n,ctypes in
            [('eec','f'),('r','f'),('z','f')]}
    for name, arr in arrs.items():
        ot.Branch(name, arr, f"{name}/{arr.typecode}")

    N=0

    for iEvt in range(t_hadrons.GetEntries()):
        t_hadrons.GetEntry(iEvt)

        h0.Fill(0.5)

        if not isGen and (t_hadrons.passesSTheta < 0.5 or t_hadrons.passesNTrkMin < 0.5 or t_hadrons.passesTotalChgEnergyMin < 0.5): continue
            
        h0.Fill(1.5)

        E = t_hadrons.Energy
        
        px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco, hp_reco, pwflag = (
            np.array(getattr(t_hadrons, attr)) for attr in ['px', 'py', 'pz', 'mass', 'charge', 'theta', 'pt', 'eta', 'phi', 'highPurity', 'pwflag']
            )
        
        if not isGen:
            sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 2.795) & (theta_reco > 0.348) & (hp_reco > 0.5)
        else:
            conversion = conversion_veto_mask(eta_reco, phi_reco, c_reco, pwflag, 0.05, 0.05)
            #sel_reco = (abs(c_reco) > 0.1) & conversion
            sel_reco = (abs(c_reco) > 0.1) & (hp_reco > 0.5)
            #sel_reco = (abs(c_reco) > 0.1) & (pt_reco > 0.2) & (theta_reco < 160*np.pi/180) & (theta_reco > 20*np.pi/180) & conversion
            mask_reco = ~conversion
            theta_conv = theta_reco
            theta_conv = theta_conv[mask_reco]

            for t in theta_conv:
                h1d["eta_conv_ele"].Fill(t)
            
        px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco = (
            arr[sel_reco] for arr in [px_reco, py_reco, pz_reco, m_reco, c_reco, theta_reco, pt_reco, eta_reco, phi_reco]
        )

        e_reco = np.sqrt(px_reco**2 + py_reco**2 + pz_reco**2 + m_reco**2)
        p3 = np.stack((px_reco, py_reco, pz_reco), axis=1)

        for i,j in itertools.combinations(range(len(p3)), 2):
            r = calcAngle(p3[i], p3[j])
            eec = e_reco[i]*e_reco[j] / (E**2)
            z   = 0.5*(1-np.cos(r))
            fill_pair(i, j, r, z, eec, c_reco)
            if False:
                r[0] = r
                z[0] = z
                eec[0] = eec
                ot.Fill()

        N += 1

    print("Processed", N, "events")
    
    #otree.Write()

    fout.cd()
    h0.Write()
    for key in h1d.keys():
        h1d[key].Write()
    for key in h2d.keys():
        h2d[key].Write()
