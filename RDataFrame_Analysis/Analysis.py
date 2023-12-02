from ROOT import RDataFrame
import ROOT
import matplotlib.pyplot as plt
import awkward as ak
import json
from utils import (
    get_lepton_preselection, 
    get_jet_preselection
)



##################  Initializing Stage  ##################


#change the redirector for the one that works for you, possible values: 
#cmsxrootd.fnal.gov for US and America, 
#xrootd-cms.infn.it for Europe and rest of Asia, 
#cms-xrd-global.cern.ch for rest of the world

#Only change the redirector and channel

year = '2017'
#redirector = 'cmsxrootd.fnal.gov'
redirector = ''
channel = "electron"


with open(f"fileset_{year}_UL_NANO.json", "r") as f:
    fileset = json.load(f)


fileste = ['redirector' + path_file for path_file in fileset]


### Reading the ROOT FIles
names = ROOT.std.vector('string')()
for n in [filename for filename in file]: names.push_back(n)
df = RDataFrame("Events", names)

df = df.Range(10000000)



##################  Preselection Stage  ##################


with open(f"{channel}_selection.json", "r") as f:
    selection_dict = json.load(f)

lepton_preselection = selection_dict["preselection"]['lepton']
jet_preselection = selection_dict["preselection"]['jet']


preselections = {
    "good_electron": get_lepton_preselection(lepton_preselection, "ele"),
    "good_muon": get_lepton_preselection(lepton_preselection, "mu"),
    "good_tau": get_lepton_preselection(lepton_preselection, "tau"),
    "good_bjet": get_jet_preselection(jet_preselection),
}

for name, preselection in preselections.items():
    df = df.Define(name, preselection)


##################  Selection Stage  ##################

selections = selection_dict["selection"]

for selection_name, selection_value,  in selections.items():
    #print(selection_value)
    df = df.Filter(selection_value, selection_name)
    
    


##################  Output Stage  ##################


if channel == 'electron':
    lepton = 'Electron'
elif channel == 'muon':
    lepton = 'Muon'

lepton = 'Electron'

df =  df.Define(f"good_{lepton}_pt", f"{lepton}_pt[good_{channel}]")\
        .Define(f"good_{lepton}_eta", f"{lepton}_eta[good_{channel}]")\
        .Define(f"good_{lepton}_phi", f"{lepton}_phi[good_{channel}]")\
        .Define("good_B_pt", "Jet_pt[good_bjet]")\
        .Define("good_B_eta", "Jet_eta[good_bjet]")\
        .Define("good_B_phi", "Jet_phi[good_bjet]")

array = ak.from_rdataframe(
        rdf = df,
        columns=(
            f"good_{lepton}_pt",
            f"good_{lepton}_phi",
            f"good_{lepton}_eta",
            "good_B_pt",
            "good_B_phi",
            "good_B_eta",
        ),
    )


ak.to_parquet(array, "example_py.parquet")

