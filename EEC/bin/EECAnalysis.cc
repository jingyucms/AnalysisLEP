#include <TH1F.h>
#include <TH2F.h>

#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>

#include "TFile.h"
#include "TSystemFile.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTree.h"
#include "TChain.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TClonesArray.h"

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"

#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"


float E_ij(float px1, float py1, float pz1, float m1, float px2, float py2, float pz2, float m2) {
    float E1 = std::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + m1 * m1);
    float E2 = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + m2 * m2);
    return E1 * E2;
}

float theta(float px1, float py1, float pz1, float px2, float py2, float pz2) {
    float dotProduct = px1 * px2 + py1 * py2 + pz1 * pz2;
    float mag1 = std::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1);
    float mag2 = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2);
    return dotProduct / (mag1 * mag2);
}

//int main(int argc, char* argv[]) {
//}

using namespace std;
int main(int argc, char* argv[]) {

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  if ( argc < 2 ) {
    cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  if( !edm::cmspybind11::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ){
    cout << " ERROR: ParametersSet 'process' is missing in your configuration file" << endl;
    exit(0);
  }
  
  // get the python configuration
  const edm::ParameterSet& process = edm::cmspybind11::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");
  fwlite::InputSource inputHandler_(process);
  fwlite::OutputFiles outputHandler_(process);

  const edm::ParameterSet& ana = process.getParameter<edm::ParameterSet>("EECAnalysis");
  bool isData_( ana.getParameter<bool>("IsData") );
  int nEvt_( inputHandler_.maxEvents() );
  
//  TChain *fChain;
//  fChain = new TChain("t");
//
//  for(unsigned int iFile=0; iFile<inputHandler_.files().size(); ++iFile){
//    string inputFile=inputHandler_.files()[iFile];
//    cout << "File: " << inputFile << endl;
//    fChain->Add(inputFile.c_str());
//  }

//  Long64_t nevents(0);
//  if (nEvt_>0){
//    nevents=nEvt_;
//  }else{
//    nevents=fChain->GetEntries();
//  }
//
//  cout << "Number of events: " << nevents << endl;

//  Float_t Energy;
//  Int_t EventNo, nParticle;
//  Float_t *px = nullptr, *py = nullptr, *pz = nullptr, *mass = nullptr;
//  Short_t *charge = nullptr;
  
//  fChain->SetBranchAddress("Energy", &Energy);
//  fChain->SetBranchAddress("EventNo", &EventNo);
//  fChain->SetBranchAddress("nParticle", &nParticle);
//  fChain->SetBranchAddress("px", &px);
//  fChain->SetBranchAddress("py", &py);
//  fChain->SetBranchAddress("pz", &pz);
//  fChain->SetBranchAddress("mass", &mass);
//  fChain->SetBranchAddress("charge", &charge);
//
//  fChain->SetMakeClass(1);

  string inputFile=inputHandler_.files()[0];
  TFile *myFile = TFile::Open(inputFile.c_str());

  TTreeReader particleReader("t", myFile);
  TTreeReaderValue<int> nParticle(   particleReader, "NParticle");
  TTreeReaderValue<int> EventNo(   particleReader, "EventNo");
  TTreeReaderArray<float> Energy(   particleReader, "Energy");
  TTreeReaderArray<float> px(      particleReader, "px");
  TTreeReaderArray<float> py(      particleReader, "py");
  TTreeReaderArray<float> pz(      particleReader, "pz");
  TTreeReaderArray<float> mass(      particleReader, "mass");
  TTreeReaderArray<Short_t> charge(  particleReader, "charge");

  TTree *_outTree;
  Float_t Weight, EEC, r;
  Int_t Event;

  string outFile=outputHandler_.file().c_str();
  cout << "Output written to: " << outFile << endl;
  TFile* OutFile = TFile::Open(outputHandler_.file().c_str(),"RECREATE");

  _outTree = new TTree("EEC", "EEC");
  _outTree->Branch("Weight",   &Weight  ,  "Weight/F");
  _outTree->Branch("EEC",   &EEC  ,  "EEC/F");
  _outTree->Branch("r",   &r  ,  "r/F");
  _outTree->Branch("Event",   &Event  ,  "Event/I");

//  int N = 0;
//  for (Long64_t entry = 0; entry < nevents; ++entry) {
//    cout << entry << endl;
//    //fChain->GetEntry(entry);
//    //fChain->Print();
//    particleReader.Next();
//    //cout << nParticle << endl;
//    //cout << typeid(*px).name() << endl;
//    float E = *Energy;
//    Event = *EventNo;
//    
//    for (int i = 0; i < *nParticle; ++i) {
//      for (int j = 0; j < *nParticle; ++j) {
//	if (i >= j) continue;
//	if (std::abs(charge[i]) < 0.01 || std::abs(charge[j]) < 0.01) continue;
//	float Eij = E_ij(px[i], py[i], pz[i], mass[i],
//			 px[j], py[j], pz[j], mass[j]);
//	float r = theta(px[i], py[i], pz[i], 
//			px[j], py[j], pz[j]);
//	
//	//h->Fill(r, Eij / (E * E));
//	EEC = Eij / (E * E);
//	r = r;
//	if (isData_) Weight = 1.;
//	else Weight = 1.;
//	_outTree -> Fill();
//      }
//    }
//    ++N;
//  }

  OutFile->cd();
  _outTree->Write();
  OutFile->Close();
}
