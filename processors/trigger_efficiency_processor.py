import json
import pickle
import numpy as np
import pandas as pd
import awkward as ak
import hist as hist2
from typing import List
from coffea import processor
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


class TriggerEfficiencyProcessor(processor.ProcessorABC):
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
            "cut_flow": {},
            "electron_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="electron_pt",
                    label=r"electron $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="electron_relIso", label="electron RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="electron_eta", label="electron $\eta$"),
                hist2.storage.Weight(),
            ),
            "muon_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="muon_pt",
                    label=r"muon $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="muon_relIso", label="muon RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="muon_eta", label="muon $\eta$"),
                hist2.storage.Weight(),
            ),
            "jet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500], 
                    name="jet_pt", 
                    label=r"bJet $p_T$ [GeV]"
                ),
                hist2.axis.Regular(50, -2.4, 2.4, name="jet_eta", label="bJet $\eta$"),
                hist2.storage.Weight(),
            ),
            "met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
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
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(
                    40, 10, 800, name="electron_met_mt", label=r"$M_T$(electron, bJet) [GeV]"
                ),
                hist2.axis.Regular(
                    40, 10, 800, name="muon_met_mt", label=r"$M_T$(muon, bJet) [GeV]"
                ),
                hist2.axis.Regular(
                    30, 0, 5, name="electron_bjet_dr", label="$\Delta R$(electron, bJet)"
                ),
                 hist2.axis.Regular(
                    30, 0, 5, name="muon_bjet_dr", label="$\Delta R$(muon, bJet)"
                ),
                hist2.storage.Weight(),
            ),
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
        self.output['cut_flow']['nevents'] = nevents
    
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
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)
        electrons = ak.firsts(events.Electron[good_electrons])
        electrons_p4 = build_p4(electrons)
        
        # muons
        good_muons = (
            (events.Muon.pt >= 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId if self._channel == "ele" else events.Muon.tightId)
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = ak.firsts(events.Muon[good_muons])
        muons_p4 = build_p4(muons)
        
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
            mod=self._yearmod,
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
        weights = Weights(nevents, storeIndividual=True)
        if self.isMC:
            # genweight
            self.output["sumw"] = ak.sum(events.genWeight)
            weights.add("genweight", events.genWeight)
            # L1prefiring
            if self._year in ("2016", "2017"):
                weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # pileup
            add_pileup_weight(
                weights=weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # b-tagging
            self._btagSF = BTagCorrector(
                wp="M", tagger="deepJet", year=self._year, mod=self._yearmod
            )
            self._btagSF.add_btag_weight(jets=events.Jet[good_bjets], weights=weights)

            # electron weights
            add_electronID_weight(
                weights=weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year, 
                mod=self._yearmod,
                wp="wp80noiso" if self._channel == "ele" else "wp90noiso"
            )
            add_electronReco_weight(
                weights=weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year,
                mod=self._yearmod,
            )
            add_electronTrigger_weight(
                weights=weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year, 
                mod=self._yearmod,
            )
            # muon weights
            add_muon_weight(
                weights=weights,
                muon=ak.firsts(events.Muon[good_muons]), 
                sf_type="id", 
                year=self._year, 
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight"
            )
            add_muon_weight(
                weights=weights, 
                muon=ak.firsts(events.Muon[good_muons]), 
                sf_type="iso", 
                year=self._year, 
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight"
            )
            add_muonTriggerIso_weight(
                weights=weights, 
                muon=ak.firsts(events.Muon[good_muons]), 
                year=self._year, 
                mod=self._yearmod,
            )

        # selections
        self.selections = PackedSelection()
        self.add_selection("trigger_ele", trigger["ele"])
        self.add_selection("trigger_mu", trigger["mu"])
        self.add_selection("lumi", lumi_mask)
        self.add_selection("metfilters", metfilters)
        self.add_selection("one_electron", n_good_electrons == 1)
        self.add_selection("one_muon", n_good_muons == 1)
        self.add_selection("ele_reliso", ele_reliso < 0.25)
        self.add_selection("mu_reliso", mu_reliso < 0.25)
        self.add_selection("deltaR", mu_bjet_dr > 0.4)
        self.add_selection("two_bjets", n_good_bjets >= 1)
        
        
        # regions
        regions = {
            "ele": {
                "numerator": [
                    "lumi",
                    "metfilters",
                    "one_electron",
                    "one_muon",
                    "two_bjets",
                    "trigger_ele", 
                    "trigger_mu",  
                ],
                "denominator": [
                    "lumi",
                    "metfilters",
                    "one_electron",
                    "one_muon",
                    "two_bjets",
                    "trigger_mu", 
                ],
            },
            "mu": {
                "numerator": [
                    "lumi",
                    "metfilters",
                    "one_electron",
                    "one_muon",
                    "ele_reliso",
                    "mu_reliso",
                    "deltaR",
                    "two_bjets",
                    "trigger_ele",
                    "trigger_mu",
                ],
                "denominator": [
                    "lumi",
                    "metfilters",
                    "one_electron",
                    "one_muon",
                    "ele_reliso",
                    "mu_reliso",
                    "deltaR",
                    "two_bjets",
                    "trigger_ele",
                ],
            },
        }
        # weights per region
        common_weights = ["genweight", "L1Prefiring", "pileup", "btagSF"]
        electron_weights = ["electronReco", "electronID"]
        muon_weights = ["muonIso", "muonId"]
        
        numerator_weights = (
            common_weights
            + electron_weights
            + muon_weights
            + ["electronTrigger", "muonTriggerIso"]
        )
        denominator_weights = common_weights + electron_weights + muon_weights
        
        weights_per_region = {
            "ele": {
                "numerator": numerator_weights,
                "denominator": denominator_weights + ["muonTriggerIso"]
            },
            "mu": {
                "numerator": numerator_weights,
                "denominator": denominator_weights + ["electronTrigger"],
            }
        }
        # filling histograms
        def fill(region: str):
            selections = regions[self._channel][region]
            cut = self.selections.all(*selections)
            
            region_weights = weights_per_region[self._channel][region]
            region_weight = weights.partial_weight(region_weights)[cut]
    
            self.output["jet_kin"].fill(
                region=region,
                jet_pt=normalize(candidatebjet.pt, cut),
                jet_eta=normalize(candidatebjet.eta, cut),
                weight=region_weight,
            )
            self.output["met_kin"].fill(
                region=region,
                met=normalize(met.pt, cut),
                met_phi=normalize(met.phi, cut),
                weight=region_weight,
            )

            self.output["electron_kin"].fill(
                region=region,
                electron_pt=normalize(electrons.pt, cut),
                electron_relIso=normalize(ele_reliso, cut),
                electron_eta=normalize(electrons.eta, cut),
                weight=region_weight,
            )
            self.output["muon_kin"].fill(
                region=region,
                muon_pt=normalize(muons.pt, cut),
                muon_relIso=normalize(mu_reliso, cut),
                muon_eta=normalize(muons.eta, cut),
                weight=region_weight,
            )
            self.output["mix_kin"].fill(
                region=region,
                electron_met_mt=normalize(mt_ele_met, cut),
                muon_met_mt=normalize(mt_mu_met, cut),
                electron_bjet_dr=normalize(ele_bjet_dr, cut),
                muon_bjet_dr=normalize(mu_bjet_dr, cut),
                weight=region_weight,
            )
           
        for region in regions[self._channel]:
            fill(region)
                
        return {dataset: self.output}

    def postprocess(self, accumulator):
        return accumulator