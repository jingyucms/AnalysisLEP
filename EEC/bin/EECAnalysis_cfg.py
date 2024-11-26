import FWCore.ParameterSet.Config as cms
import os

#####################################################################################################

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring(
        '/eos/user/z/zhangj/ALEPH/Samples/ALEPH/LEP1Data1994P1_recons_aftercut-MERGED.root'
        #infiles
        # 'root://eoscms//eos/cms/store/cmst3/user/hinzmann/fastsim/herwigpp_qcd_m1400___Sep4/PFAOD_1.root'
                              ), ## mandatory
    maxEvents   = cms.int64(-1),                             ## optional
    outputEvery = cms.uint32(10),                            ## optional
)
    
process.fwliteOutput = cms.PSet(
    ## fileName  = cms.string('analyzeChi_14TeV_m'+massCut+'inf_CI20TEV.root'),  ## mandatory
    fileName  = cms.string('test.root'),  ## mandatory
)

process.EECAnalysis = cms.PSet(
    ## input specific for this analyzer
    IsData = cms.bool(True)
    )
