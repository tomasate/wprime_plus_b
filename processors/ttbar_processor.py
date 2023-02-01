import os
import json
import pickle
import correctionlib
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa
import hist as hist2
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Union
from typing import Type
from coffea import util
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection
from .utils import normalize, pad_val, build_p4
from .corrections import (
    BTagCorrector,
    add_pileup_weight,
    add_electronID_weight,
    add_electronReco_weight,
    add_electronTrigger_weight,
    add_muon_weight,
    add_muonTriggerIso_weight,
    get_met_corrections
)

class TTbarControlRegionProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        channel: str = "ele",
        output_location: str = "",
        dir_name: str = "",
    ):
        self._year = year
        self._yearmod = yearmod
        self._channel = channel
        self._output_location = output_location
        self._dir_name = dir_name

        # open triggers
        with open("/home/cms-jovyan/wprime_plus_b/data/triggers.json", "r") as f:
            self._triggers = json.load(f)[self._year]

        # open btagDeepFlavB
        with open("/home/cms-jovyan/wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]

        # open met filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open(
            "/home/cms-jovyan/wprime_plus_b/data/metfilters.json", "rb"
        ) as handle:
            self._metfilters = json.load(handle)[self._year]

        # open lumi masks
        with open("/home/cms-jovyan/wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
        
        # output histograms
        self.make_output = lambda: {
            "sumw": 0,
            "cut_flow":{},
            "electron_kin": hist2.Hist(
                
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="electron_pt",
                    label=r"electron $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="electron_relIso", label="electron RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="electron_eta", label="electron $\eta$"),
                hist2.axis.Regular(50,-4.0, 4.0, name='electron_phi', label="electron $\phi$"),
                hist2.storage.Weight(),
            ),
            "muon_kin": hist2.Hist(
                
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="muon_pt",
                    label=r"muon $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="muon_relIso", label="muon RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="muon_eta", label="muon $\eta$"),
                hist2.axis.Regular(50,-4.0, 4.0, name='muon_phi', label="muon phi"),
                hist2.storage.Weight(),
            ),
            "jet_kin": hist2.Hist(
                
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500], 
                    name="jet_pt", 
                    label=r"bJet $p_T$ [GeV]"
                ),
                hist2.axis.Regular(50, -2.4, 2.4, name="jet_eta", label="bJet $\eta$"),
                hist2.axis.Regular(50,-4.0, 4.0, name='jet_phi'),
                hist2.storage.Weight(),
            ),
            "met_kin": hist2.Hist(
               
                hist2.axis.Variable(
                    [50, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="met",
                    label=r"$p_T^{miss}$ [GeV]",
                ),
                hist2.axis.Regular(
                    50, -4.0, 4.0, name="met_phi", label=r"$\phi(p_T^{miss})$"
                ),
                hist2.storage.Weight(),
            ),
            "mix_kin": hist2.Hist(
                
                hist2.axis.Regular(
                    40, 10, 800, name="electron_met_mt", label=r"$M_T$(electron, bJet) [GeV]"
                ),
                hist2.axis.Regular(
                    40, 10, 800, name="muon_met_mt", label=r"$M_T$(muon, bJet) [GeV]"
                ),
                #hist2.axis.Regular(
                #    30, 0, 5, name="electron_bjet_dr", label="$\Delta R$(electron, bJet)"
                #),
                 hist2.axis.Regular(
                    30, 0, 5, name="muon_bjet_dr", label="$\Delta R$(muon, bJet)"
                ),
                hist2.storage.Weight(),
            )
            
        }
        
    def add_selection(
        self,
        name: str,
        sel: ak.Array,
    ) -> None:
        """
        Adds selection to PackedSelection object and the cutflow dictionary
        taken from: github.com/cmantill/boostedhiggs/blob/main/boostedhiggs/hwwprocessor.py
        """
        self.selections.add(name, sel)
        selection = self.selections.all(*self.selections.names)
        '''if self.isMC:
            weight = self.weights.weight()
            self.output['cut_flow'][name] = float(weight[selection].sum())'''
        #else:
        self.output['cut_flow'][name] = np.sum(selection)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.output = self.make_output()
        self.output['cut_flow']['nevents']= nevents
        
    
        # luminosity
        if not self.isMC:
            lumi_mask = self._lumi_mask[self._year](events.run, events.luminosityBlock)
        else:
            lumi_mask = np.ones(len(events), dtype="bool")

        # MET filters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # triggers
        trigger = {}
        for ch in ["ele", "mu"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._triggers[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]

        # electrons
        good_electrons = (
            (events.Electron.pt >= 30)
            & (np.abs(events.Electron.eta) < 2.4)
            & (
                (np.abs(events.Electron.eta) < 1.44)
                | (np.abs(events.Electron.eta) > 1.57)
            )
            & (
                events.Electron.mvaFall17V2Iso_WP80
                if self._channel == "ele"
                else events.Electron.mvaFall17V2Iso_WP90
            )
            & (events.Electron.pfRelIso04_all < 0.25 if hasattr(events.Electron, "pfRelIso04_all") else events.Electron.pfRelIso03_all < 0.25)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)
        electrons = ak.firsts(events.Electron[good_electrons])
        electrons_p4 = build_p4(electrons)
        
        # muons
        good_muons = (
            (events.Muon.pt >= 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId if self._channel == "ele" else events.Muon.tightId)
            & (events.Muon.pfRelIso04_all < 0.25 if hasattr(events.Muon, "pfRelIso04_all") else events.Muon.pfRelIso03_all < 0.25)
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = ak.firsts(events.Muon[good_muons])
        muons_p4 = build_p4(muons)
        
        #Tau
        good_taus=(
            (events.Tau.idDeepTau2017v2p1VSjet > 8)
            & (events.Tau.idDeepTau2017v2p1VSe > 8)
            & (events.Tau.idDeepTau2017v2p1VSmu > 1)
            & (np.abs(events.Tau.eta) < 2.3)
            & (events.Tau.pt > 20)
            & (events.Tau.dz < 0.2))
        n_good_taus = ak.sum(good_taus, axis=1)
        taus = ak.firsts(events.Tau[good_taus])
        taus_p4 = build_p4(taus)
        
        # b-jets
        good_bjets = (
            (events.Jet.pt >= 20)
            & (events.Jet.jetId == 6)
            & (events.Jet.puId == 7)
            & (events.Jet.btagDeepFlavB > self._btagDeepFlavB)
            & (np.abs(events.Jet.eta) < 2.4)
        )
        n_good_bjets = ak.sum(good_bjets, axis=1)
        candidatebjet = ak.firsts(events.Jet[good_bjets])
        
        # missing energy
        met = events.MET
        met["pt"], met["phi"] = get_met_corrections(
            year=self._year,
            is_mc=self.isMC,
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvs,
            mod=self._yearmod
        )
        
        # relative Iso
        ele_reliso = (
            electrons.pfRelIso04_all
            if hasattr(electrons, "pfRelIso04_all")
            else electrons.pfRelIso03_all
        )
        
        mu_reliso = (
            muons.pfRelIso04_all
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all
        )
        
        # lepton-bjet delta R
        ele_bjet_dr = candidatebjet.delta_r(electrons_p4)
        mu_bjet_dr = candidatebjet.delta_r(muons_p4)

        # lepton-MET transverse mass
        mt_ele_met = np.sqrt(
            2.0
            * electrons_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(electrons_p4.delta_phi(met)))
        )
        mt_mu_met = np.sqrt(
            2.0
            * muons_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(muons_p4.delta_phi(met)))
        )
            
        # weights
        self.weights = Weights(nevents, storeIndividual=True)
        if self.isMC:
            # genweight
            self.output["sumw"] = ak.sum(events.genWeight)
            self.weights.add("genweight", events.genWeight)
            # L1prefiring
            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # pileup
            add_pileup_weight(
                weights=self.weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # b-tagging
            self._btagSF = BTagCorrector(
                wp="M", tagger="deepJet", year=self._year, mod=self._yearmod
            )
            self._btagSF.add_btag_weight(jets=events.Jet[good_bjets], weights=self.weights)

            # electron weights
            add_electronID_weight(
                weights=self.weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year, 
                mod=self._yearmod,
                wp="wp80noiso" if self._channel == "ele" else "wp90noiso"
            )
            add_electronReco_weight(
                weights=self.weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year,
                mod=self._yearmod,
            )
            if self._channel == 'ele':
                add_electronTrigger_weight(
                    weights=self.weights, 
                    electron=ak.firsts(events.Electron[good_electrons]), 
                    year=self._year, 
                    mod=self._yearmod,
                )
            # muon weights
            add_muon_weight(
                weights=self.weights,
                muon=ak.firsts(events.Muon[good_muons]), 
                sf_type="id", 
                year=self._year, 
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight"
            )
            add_muon_weight(
                weights=self.weights, 
                muon=ak.firsts(events.Muon[good_muons]), 
                sf_type="iso", 
                year=self._year, 
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight"
            )
            if self._channel == 'mu':     
                add_muonTriggerIso_weight(
                    weights=self.weights, 
                    muon=ak.firsts(events.Muon[good_muons]), 
                    year=self._year, 
                    mod=self._yearmod,
                )
        
        # selections
        self.selections = PackedSelection()
        self.add_selection("lumi", lumi_mask)
        self.add_selection("metfilters", metfilters)
        if self._channel == 'ele':
            self.add_selection("trigger_ele", trigger["ele"])
            self.add_selection("good_electron", n_good_electrons == 1)
            self.add_selection("good_muon", n_good_muons == 0)
        elif self._channel == 'mu':
            self.add_selection("trigger_mu", trigger["mu"])
            self.add_selection("good_electron", n_good_electrons == 0)
            self.add_selection("good_muon", n_good_muons == 1)
            self.add_selection("deltaR", mu_bjet_dr > 0.4)
        self.add_selection("good_tau", n_good_taus == 0)
        self.add_selection("met_pt", met.pt > 50)
        self.add_selection("two_bjets", n_good_bjets == 2)
       
    
        # regions
        regions = {
            "ele": ["lumi",
                    "metfilters",
                    "trigger_ele", 
                    "good_electron",
                    "good_muon",
                    "good_tau",
                    "met_pt",
                    "two_bjets",             
                ],
            
            "mu": ["lumi",
                   "metfilters",
                   "trigger_mu",
                   "good_muon",
                   "deltaR",
                   "good_electron",
                   "good_tau",
                   "met_pt",
                   "two_bjets"              
                ]
        }
       
       
        # filling histograms
        selections = regions[self._channel]
        cut = self.selections.all(*selections)
        region_weight = self.weights.weight()[cut]
        
        def fill():
            self.output["jet_kin"].fill(
                jet_pt=normalize(candidatebjet.pt, cut),
                jet_eta=normalize(candidatebjet.eta, cut),
                jet_phi=normalize(candidatebjet.phi,cut),
                weight=region_weight,
            )
            self.output["met_kin"].fill(
                met=normalize(met.pt, cut),
                met_phi=normalize(met.phi, cut),
                weight=region_weight,
            )
            self.output["electron_kin"].fill(
                electron_pt=normalize(electrons.pt, cut),
                electron_relIso=normalize(ele_reliso, cut),
                electron_eta=normalize(electrons.eta, cut),
                electron_phi=normalize(electrons.phi, cut),
                weight=region_weight,
            )
            self.output["muon_kin"].fill(            
                muon_pt=normalize(muons.pt, cut),
                muon_relIso=normalize(mu_reliso, cut),
                muon_eta=normalize(muons.eta, cut),
                muon_phi=normalize(muons.phi, cut),
                weight=region_weight,
            )
            self.output["mix_kin"].fill(    
                electron_met_mt=normalize(mt_ele_met, cut),
                muon_met_mt=normalize(mt_mu_met, cut),
                #electron_bjet_dr=normalize(ele_bjet_dr, cut),
                muon_bjet_dr=normalize(mu_bjet_dr, cut),
                weight=region_weight,
            )
        
        fill()
                
        return {dataset: self.output}

    def postprocess(self, accumulator):
        return accumulator