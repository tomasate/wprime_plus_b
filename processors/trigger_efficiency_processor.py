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

        # dataset names per year
        if year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }

        self.make_output = lambda: {
            "sumw": 0,
            "lepton_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="lep_pt",
                    label=r"lep $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="lep_miniIso", label="lep miniIso"),
                hist2.axis.Regular(25, 0, 1, name="lep_relIso", label="lep RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="lep_eta", label="lep $\eta$"),
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
                hist2.storage.Weight(),
            ),
            "mix_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(
                    40, 10, 800, name="lep_met_mt", label=r"$M_T$(lep, bJet) [GeV]"
                ),
                hist2.axis.Regular(
                    30, 0, 5, name="lep_bjet_dr", label="$\Delta R$(lep, bJet)"
                ),
                hist2.storage.Weight(),
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = len(events)

        output = self.make_output()
        self.isMC = hasattr(events, "genWeight")

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

        # muons
        # mediumId OR tightId?
        # DO WE NEED LOOSE MUONS?
        good_muons = (
            (events.Muon.pt >= 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId if self._channel == "ele" else events.Muon.tightId)
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # get candidate lepton
        if self._channel == "ele":
            candidatelep = events.Electron[good_electrons]
        if self._channel == "mu":
            candidatelep = events.Muon[good_muons]
        candidatelep = ak.firsts(candidatelep)
        candidatelep_p4 = build_p4(candidatelep)

        # reliso for candidate lepton
        lep_reliso = (
            candidatelep.pfRelIso04_all
            if hasattr(candidatelep, "pfRelIso04_all")
            else candidatelep.pfRelIso03_all
        )
        # miniso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all

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

        # lepton-bjet delta R
        lep_bjet_dr = candidatebjet.delta_r(candidatelep_p4)

        # MET
        met = events.MET

        # lepton-MET transverse mass
        mt_lep_met = np.sqrt(
            2.0
            * candidatelep_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # weights
        weights = Weights(nevents, storeIndividual=True)
        if self.isMC:
            # genweight
            output["sumw"] = ak.sum(events.genWeight)
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

            # lepton weights
            add_electronID_weight(
                weights=weights, 
                electron=ak.firsts(events.Electron[good_electrons]), 
                year=self._year, 
                mod=self._yearmod,
                wp="wp80iso" if self._channel == "ele" else "wp90iso"
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
        selection = PackedSelection()
        one_lepton = {
            "ele": (n_good_muons == 1) & (n_good_electrons == 1),
            "mu": (n_good_muons == 1) & (n_good_electrons == 1),
        }
        selection.add("trigger_ele", trigger["ele"])
        selection.add("trigger_mu", trigger["mu"])
        selection.add("lumi", lumi_mask)
        selection.add("metfilters", metfilters)
        selection.add("two_bjets", n_good_bjets >= 1)
        selection.add("one_lepton", one_lepton[self._channel])
        selection.add("good_electron", ak.firsts(good_electrons))
        selection.add("good_muon", ak.firsts(good_muons))
        selection.add("delta_r", lep_bjet_dr > 0.4)

        # regions
        regions = {
            "ele": {
                "numerator": [
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_lepton",
                    "trigger_ele",
                    "trigger_mu",
                    "good_electron",
                    "good_muon"
                ],
                "denominator": [
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_lepton",
                    "trigger_mu",
                    "good_electron",
                    "good_muon"
                ],
            },
            "mu": {
                "numerator": [
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_lepton",
                    "trigger_ele",
                    "trigger_mu",
                    "good_electron",
                    "good_muon"
                ],
                "denominator": [
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_lepton",
                    "trigger_ele",
                    "good_electron",
                    "good_muon"
                ],
            },
        }

        def fill(region: str):
            selections = regions[self._channel][region]
            cut = selection.all(*selections)

            output["lepton_kin"].fill(
                region=region,
                lep_pt=normalize(candidatelep.pt, cut),
                lep_miniIso=normalize(lep_miso, cut),
                lep_relIso=normalize(lep_reliso, cut),
                lep_eta=normalize(candidatelep.eta, cut),
                weight=weights.weight()[cut],
            )
            output["jet_kin"].fill(
                region=region,
                jet_pt=normalize(candidatebjet.pt, cut),
                jet_eta=normalize(candidatebjet.eta, cut),
                weight=weights.weight()[cut],
            )
            output["met_kin"].fill(
                region=region,
                met=normalize(met.pt, cut),
                weight=weights.weight()[cut],
            )
            output["mix_kin"].fill(
                region=region,
                lep_met_mt=normalize(mt_lep_met, cut),
                lep_bjet_dr=normalize(lep_bjet_dr, cut),
                weight=weights.weight()[cut],
            )

        for region in regions[self._channel]:
            fill(region)

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
