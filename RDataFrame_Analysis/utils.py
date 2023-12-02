import uproot
import awkward as ak

def create_tree_from_tree(in_path, out_path, in_tree_name = 'Delphes',
                          variables = ['Jet/Jet.PT',
                                       'Jet/Jet.Eta',
                                       'Jet/Jet.Phi',
                                       'Jet/Jet.Mass',
                                       'Jet/Jet.Flavor',
                                       'Jet/Jet.BTag',
                                       'Jet/Jet.TauTag',
                                       'Jet/Jet.Charge',
                                       'Jet_size',
                                       'Muon/Muon.PT',
                                       'Muon/Muon.Eta',
                                       'Muon/Muon.Phi',
                                       'Muon/Muon.Charge',
                                       'Muon_size',
                                       'Photon/Photon.PT',
                                       'Photon/Photon.Eta',
                                       'Photon/Photon.Phi',
                                       'Photon/Photon.E',
                                       'Photon_size',
                                       'Electron/Electron.PT',
                                       'Electron/Electron.Eta',
                                       'Electron/Electron.Phi',
                                       'Electron/Electron.Charge',
                                       'Electron_size',
                                       'MissingET/MissingET.MET',
                                       'MissingET/MissingET.Eta',
                                       'MissingET/MissingET.Phi'
                                      ]
                         ):

    """
    This function creates a new ROOT TTree file by extracting specific variables from an input TTree file and storing them in a new structure.
    
    Parameters:
    - in_path (str): The path to the input TTree file.
    - out_path (str): The path to the output TTree file to be created.
    - in_tree_name (str, optional): The name of the input TTree in the input file. Default is 'Delphes'.
    - variables (list, optional): A list of variable paths to extract from the input TTree. Default is an empty list.
    
    Returns:
    None
    """

    tree = uproot.open(in_path)[in_tree_name]
    arrays = tree.arrays(filter_name = variables)
                                           
    jets = ak.zip({name[4:]: array for name, array in zip(ak.fields(arrays), ak.unzip(arrays)) if name.startswith("Jet")})
    muons = ak.zip({name[5:]: array for name, array in zip(ak.fields(arrays), ak.unzip(arrays)) if name.startswith("Muon")})
    electrons = ak.zip({name[9:]: array for name, array in zip(ak.fields(arrays), ak.unzip(arrays)) if name.startswith("Electron")})
    photons = ak.zip({name[7:]: array for name, array in zip(ak.fields(arrays), ak.unzip(arrays)) if name.startswith("Photon")})
    MET = ak.zip({name[10:]: array for name, array in zip(ak.fields(arrays), ak.unzip(arrays)) if name.startswith("MissingET")})


    tree.close()
    outfile = uproot.recreate(out_path)
    outfile["Events"] = {"Jet": jets,
                         "Muon": muons,
                         "Electrons": electrons,
                         "Photons" : photons,
                         "MET" : MET}
    outfile.close()


def my_filtering_function(pair):
    #unwanted_value_type = str
    key, value = pair
    if type(value) == str:
        return True  # filter pair out of the dictionary
    else:
        return False  # keep pair in the filtered dictionary

def get_lepton_preselection(lepton_preselection: dict, flavour: str):
    """
    Return preselection mask string for leptons

    Parameters:
    -----------
        lepton_preselection: dict with lepton preselection
        flavour: lepton flavour {"ele", "mu", "tau"}
    """
    return " && ".join(
        [
            lep_presel[flavour]
            for lep_presel in lepton_preselection.values()
            if lep_presel[flavour]
        ]
    )

def get_jet_preselection(jet_preselection: dict):
    """
    Return preselection mask string for jets

    """
    return " && ".join(
        [jet_presel for jet_presel in jet_preselection.values()]
    )

def gInterpreter_DeltaPhi():
    """
    Return a string with the ROOT function for normalize an Angle between -pi and pi
    Use: If df is an RDataFrame with an angle Difference between to objects,
    that difference is > pi and/or < -pi and that variable is in a column called DPhiCol (or what ever it is):
    

        text = gInterpreter_DeltaPhi()
        ROOT.gInterpreter.Declare(text)
        df_new = df.Define("DPhi_Col_Norm", "Norma(DphiCol)")
    
    """

    
    string = """
          auto Norma = [](ROOT::VecOps::RVec<float> phi){  
          for (int i = 0; i < phi.size(); i++){
              if (phi[i] >  M_PI){
                  phi[i] -= 2 * M_PI;
                  }
              if (phi[i] < - M_PI){
                  phi[i] += 2 * M_PI;
                  }
              }
          return phi;
          };
          """
    return string

def gInterpreter_TrMass(object = 'good_Electron'):
    """
    Returns the text for the Defining column in a RDataFrame of the transverse mass of an object.
    
    Requires a 4 columns with no empty values called:
        1. <object>_pt
        2. <object>_phi
        3. MET_pt
        4. MET_phi

    Use:
        text_tr_mass_mu = gInterpreter_TrMass('good_Muon')
        df_new = df.Define("Muon_tr_mass", text_tr_mass_mu)
    """

    text = f"sqrt(2 * {object}_pt * MET_pt * (1 - cos(MET_phi - {object}_phi)))"
    return text

