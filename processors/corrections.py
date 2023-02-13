import json
import correctionlib
import numpy as np
import os
import awkward as ak
from typing import Type
from coffea import util
from coffea.analysis_tools import Weights

loc_base = os.environ["PWD"]
# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"

# summary of pog scale factors: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btag": ["BTV", "btagging.json.gz"],
    "met": ["JME", "met.json.gz"],
}

POG_YEARS = {
    "2016": "2016postVFP_UL",
    "2016APV": "2016preVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
}

TAGGER_BRANCH = {
    "deepJet": "btagDeepFlavB",
    "deepCSV": "btagDeep",
}


def get_pog_json(json_name: str, year: str) -> str:
    """
    returns the path to the pog json file

    Parameters:
    -----------
        json_name:
            json name {'muon', 'electron', 'pileup', 'btag'}
        year:
            dataset year {'2016', '2017', '2018'}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{POG_YEARS[year]}/{pog_json[1]}"


# ----------------------------------
# pileup scale factors
# -----------------------------------
def add_pileup_weight(weights: Type[Weights], year: str, mod: str, nPU: ak.Array):
    """
    add pileup scale factor

    Parameters:
    -----------
        weights:
            Weights object from coffea.analysis_tools
        year:
            dataset year {'2016', '2017', '2018'}
        mod:
            year modifier {"", "APV"}
        nPU:
            number of true interactions (events.Pileup.nPU)
    """
    # correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="pileup", year=year + mod)
    )

    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}
    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

    # add weights
    weights.add(
        name="pileup",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


# ----------------------------------
# b-tagging scale factors
# -----------------------------------
class BTagCorrector:
    def __init__(
        self,
        sf: str = "comb",
        wp: str = "M",
        tagger: str = "deepJet",
        year: str = "2017",
        mod: str = "",
    ):
        """
        BTag corrector object

        Parameters:
        -----------
            sf:
                scale factors to use (mujets or comb)
            wp:
                worging point {'L', 'M', 'T'}
            tagger:
                tagger {'deepJet', 'deepCSV'}
            year:
                dataset year {'2016', '2017', '2018'}
            mod:
                year modifier {"", "APV"}
        """
        self._sf = sf
        self._year = year + mod
        self._tagger = tagger
        self._wp = wp
        self._branch = TAGGER_BRANCH[tagger]

        # btag working points (only for deepJet)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        with open(f"{loc_base}/data/btagWPs.json", "rb") as handle:
            btagWPs = json.load(handle)
        self._btagwp = btagWPs[tagger][year + mod][wp]

        # correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="btag", year=year + mod)
        )

    def btag_SF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset[f"{self._tagger}_{self._sf}"].evaluate(
            syst,
            self._wp,
            np.array(j.hadronFlavour),
            np.array(abs(j.eta)),
            np.array(j.pt),
        )
        return ak.unflatten(sf, nj)

    def add_btag_weight(self, jets: ak.Array, weights: Type[Weights]):
        """
        add b-tagging scale factor

        Parameters:
        -----------
            jets:
                jets selected in your analysis
            weights:
                Weights object from coffea.analysis_tools
        """
        # bjets (hadron flavor definition: 5=b, 4=c, 0=udsg)
        bjets = jets[(jets.hadronFlavour > 0) & (abs(jets.eta) < 2.5)]

        # b-tag nominal scale factors
        bSF = self.btag_SF(bjets, "central")

        # combine eff and SF as tagged SF * untagged SF
        nominal_weight = ak.prod(bSF, axis=-1)

        # add nominal weight
        weights.add(name="btagSF", weight=nominal_weight)


# ----------------------------------
# lepton scale factors
# -----------------------------------
#
# Electron
#    - ID: wp80noiso?
#    - Recon: RecoAbove20?
#    - Trigger: ?
#
# working points: (Loose, Medium, RecoAbove20, RecoBelow20, Tight, Veto, wp80iso, wp80noiso, wp90iso, wp90noiso)
#
def add_electronID_weight(
    weights: Type[Weights],
    electron: ak.Array,
    year: str,
    mod: str = "",
    wp: str = "wp80noiso",
):
    """
    add electron identification scale factor

    Parameters:
    -----------
        weights:
            Weights object from coffea.analysis_tools
        electron:
            Electron collection
        year:
            Year of the dataset {'2016', '2017', '2018'}
        mod:
            Year modifier {'', 'APV'}
        wp:
            Working point {'Loose', 'Medium', 'Tight', 'wp80iso', 'wp80noiso', 'wp90iso', 'wp90noiso'}
    """
    # correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="electron", year=year + mod)
    )
    # electron pseudorapidity range: (-inf, inf)
    electron_eta = np.array(ak.fill_none(electron.eta, 0.0))

    # electron pt range: [10, inf)
    electron_pt = np.array(ak.fill_none(electron.pt, 0.0))
    electron_pt = np.clip(
        electron_pt, 10.0, 499.999
    )  # potential problems with pt > 500 GeV

    # remove _UL from year
    year = POG_YEARS[year + mod].replace("_UL", "")

    # scale factors
    values = {}
    values["nominal"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sf", wp, electron_eta, electron_pt
    )
    values["up"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sfup", wp, electron_eta, electron_pt
    )
    values["down"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sfdown", wp, electron_eta, electron_pt
    )

    weights.add(
        name=f"electronID",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


def add_electronReco_weight(
    weights: Type[Weights],
    electron: ak.Array,
    year: str,
    mod: str = "",
):
    """
    add electron reconstruction scale factor

    Parameters:
    -----------
        weights:
            Weights object from coffea.analysis_tools
        electron:
            Electron collection
        year:
            Year of the dataset {'2016', '2017', '2018'}
        mod:
            Year modifier {'', 'APV'}
    """
    # correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="electron", year=year + mod)
    )
    # electron pseudorapidity range: (-inf, inf)
    electron_eta = np.array(ak.fill_none(electron.eta, 0.0))

    # electron pt range: (20, inf)
    electron_pt = np.array(ak.fill_none(electron.pt, 0.0))
    electron_pt = np.clip(
        electron_pt, 20.1, 499.999
    )  # potential problems with pt > 500 GeV

    # remove _UL from year
    year = POG_YEARS[year + mod].replace("_UL", "")

    # scale factors
    values = {}
    values["nominal"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sf", "RecoAbove20", electron_eta, electron_pt
    )
    values["up"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sfup", "RecoAbove20", electron_eta, electron_pt
    )
    values["down"] = cset["UL-Electron-ID-SF"].evaluate(
        year, "sfdown", "RecoAbove20", electron_eta, electron_pt
    )

    weights.add(
        name=f"electronReco",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


def add_electronTrigger_weight(
    weights: Type[Weights],
    electron: ak.Array,
    year: str,
    mod: str = "",
):
    trigger_corrections = {
        "2016APV": f"{loc_base}/data/electron_trigger_2016preVFP_UL.json",
        "2016": f"{loc_base}/data/electron_trigger_2016postVFP_UL.json",
        "2017": f"{loc_base}/data/electron_trigger_2017_UL.json",
        "2018": f"{loc_base}/data/electron_trigger_2018_UL.json",
    }
    # correction set
    cset = correctionlib.CorrectionSet.from_file(trigger_corrections[year + mod])

    # electron pt
    electron_pt = np.array(ak.fill_none(electron.pt, 0.0))
    electron_pt = np.clip(electron_pt, 10, 499.999)

    # electron pseudorapidity
    electron_eta = np.array(ak.fill_none(electron.eta, 0.0))
    electron_eta = np.clip(electron_eta, -2.499, 2.499)

    # scale factors (only nominal)
    values = {}
    values["nominal"] = cset["UL-Electron-Trigger-SF"].evaluate(
        electron_eta, electron_pt
    )
    weights.add(
        name=f"electronTrigger",
        weight=values["nominal"],
    )


# Muon
#
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018
#
#    - ID: medium prompt ID NUM_MediumPromptID_DEN_TrackerMuon?
#    - Iso: LooseRelIso with mediumID (NUM_LooseRelIso_DEN_MediumID)?
#    - Trigger iso:
#          2016: for IsoMu24 (and IsoTkMu24?) NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight?
#          2017: for isoMu27 NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight?
#          2018: for IsoMu24 NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight?
#
def add_muon_weight(
    weights: Type[Weights],
    muon: ak.Array,
    sf_type: str,
    year: str,
    mod: str = "",
    wp: str = "tight",
):
    """
    add muon ID (TightID) or Iso (LooseRelIso with mediumID) scale factors

    Parameters:
    -----------
        weights:
            Weights object from coffea.analysis_tools
        muon:
            Muon collection
        sf_type:
            Type of scale factor {'id', 'iso'}
        year:
            Year of the dataset {'2016', '2017', '2018'}
        mod:
            Year modifier {'', 'APV'}
        wp:
            Working point {'medium', 'tight'}
    """
    # correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="muon", year=year + mod)
    )
    # muon absolute pseudorapidity range: [0, 2.4)
    muon_eta = np.abs(np.array(ak.fill_none(muon.eta, 0.0)))
    muon_eta = np.clip(muon_eta, 0.0, 2.399)

    # muon pt range: [15, 120)
    muon_pt = np.array(ak.fill_none(muon.pt, 0.0))
    muon_pt = np.clip(muon_pt, 15.0, 119.999)

    # scale factors
    sfs_keys = {
        "id": "NUM_TightID_DEN_TrackerMuons"
        if wp == "tight"
        else "NUM_MediumPromptID_DEN_TrackerMuons",
        "iso": "NUM_LooseRelIso_DEN_TightIDandIPCut"
        if wp == "tight"
        else "NUM_LooseRelIso_DEN_MediumID",
    }

    values = {}
    values["nominal"] = cset[sfs_keys[sf_type]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "sf"
    )
    values["up"] = cset[sfs_keys[sf_type]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "systup"
    )
    values["down"] = cset[sfs_keys[sf_type]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "systdown"
    )
    weights.add(
        name=f"muon{sf_type.capitalize()}",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


def add_muonTriggerIso_weight(
    weights: Type[Weights],
    muon: ak.Array,
    year: str,
    mod: str = "",
):
    """
    add muon Trigger Iso (IsoMu24 or IsoMu27) scale factors

    Parameters:
    -----------
        weights:
            Weights object from coffea.analysis_tools
        muon:
            Muon collection
        year:
            Year of the dataset {'2016', '2017', '2018'}
        mod:
            Year modifier {'', 'APV'}
    """
    # correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="muon", year=year + mod)
    )
    # muon absolute pseudorapidity range: [0, 2.4)
    muon_eta = np.abs(np.array(ak.fill_none(muon.eta, 0.0)))
    muon_eta = np.clip(muon_eta, 0.0, 2.399)

    # muon pt range: [29, 200)
    muon_pt = np.array(ak.fill_none(muon.pt, 0.0))
    muon_pt = np.clip(muon_pt, 29.0, 199.999)

    # scale factors
    sfs_keys = {
        "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
        "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
    }

    values = {}
    values["nominal"] = cset[sfs_keys[year]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "sf"
    )
    values["up"] = cset[sfs_keys[year]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "systup"
    )
    values["down"] = cset[sfs_keys[year]].evaluate(
        POG_YEARS[year + mod], muon_eta, muon_pt, "systdown"
    )
    weights.add(
        name="muonTriggerIso",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


# ----------------------------------
# met phi modulation
# -----------------------------------
def get_met_corrections(
    year: str,
    is_mc: bool,
    met_pt: ak.Array,
    met_phi: ak.Array,
    npvs: ak.Array,
    mod: str = "",
):
    """
    return corrected MET pt and phi arrays

    Parameters:
    -----------
        year:
            Year of the dataset {'2016', '2017', '2018'}
        is_mc:
            If dataset is MC {True, False}
        met_pt:
            MET transverse momentum
        met_phi:
            MET azimuthal angle
        npvs:
            Total number of reconstructed primary vertices
        mod:
            Year modifier {'', 'APV'}
    """
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="met", year=year)
    )
    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt = np.clip(met_pt, 0.0, 6499.0)
    met_phi = np.clip(met_phi, -3.5, 3.5)

    run_ranges = {
        "2016APV": [272007, 278771],
        "2016": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
    }

    data_kind = "mc" if is_mc else "data"
    run = np.random.randint(run_ranges[year][0], run_ranges[year][1], size=len(met_pt))

    try:
        corrected_met_pt = cset[f"pt_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), npvs.to_numpy(), run
        )
        corrected_met_phi = cset[f"phi_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), npvs.to_numpy(), run
        )

        return corrected_met_pt, corrected_met_phi
    except:
        return met_pt, met_phi
