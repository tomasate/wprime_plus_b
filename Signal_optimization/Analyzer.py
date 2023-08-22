import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward
import mplhep as hep
#import ROOT
from array import array
hep.style.use("CMS")



class RootTreeReader:

    """ 
    Read data from a ROOT TTree 
    Parameters:
    path : string
        Path to the ROOT file
    tree_name : string (default=Delphes)
        Name of the ROOT TTree
    Attributes:
    tree: Root TTree 
    """

    def __init__(self, path: str, tree_name: str = "Delphes"):
        self.tree = uproot.open(path)[tree_name]


    def get_branches(self, branches = ["MissingET.MET",
                                       "MissingET.Eta",
                                       "MissingET.Phi",
                                       "Jet.PT",
                                       "Jet.Eta",
                                       "Jet.Phi",
                                       "Jet.Mass",
                                       "Jet.TauTag",
                                       "Jet.BTag",
                                       "Jet_size"], max_elements=4):
        """
        returns a DataFrame with branches as features
        branches : array-like
          branches to load from the ROOT tree
        max_elements : int (default=4)
          maximum number of elements to load from jagged arrays
        """   
        self._max_elements = max_elements
        self._df = pd.DataFrame(index=range(self.tree.num_entries))

        for branch in branches:
            self._join_branch(branch)

        return self._set_columns_names(self._df)


    def _join_branch(self, branch):
        """joins a branch to self._df"""
        df = self.tree.arrays(branch, library="pd")

        if "." in branch:
            if len(df) > len(df.groupby(level=0).size()):
                self._add_jagged_branch(df, branch)
            else:
                self._add_branch(df, branch)
        else:
            self._add_branch(df, branch)


    def _add_branch(self, df, branch: str):
        """adds a non-jagged branch to self.df"""
        self._df[branch] = self.tree[branch].array(library="pd").values


    def _add_jagged_branch(self, df, branch):
        """adds a jagged branch to self.df"""
        df = df.unstack().iloc[:,:self._max_elements]
        df.columns = ["{0}{1}".format(branch, i) for i in range(self._max_elements)]
        self._df = self._df.join(df)

    @staticmethod
    def _set_columns_names(df):
        df.columns = df.columns.str.lower().str.replace(".","_")
        return df


def build_df(path):
    """
    Generates a Dataframe from the root in "path"
    """
    reader = RootTreeReader(path)
    df = reader.get_branches()
    df['n_tau'] = np.sum(df.loc[:,"jet_tautag0":"jet_tautag3"],axis = 1)
    df['n_b'] = np.sum(df.loc[:,"jet_btag0":"jet_btag3"],axis = 1)
    return df


def tau_cut(df, val = 0):
    mask = (df.n_tau > val) & (df.jet_tau_index1 >= 0)
    return df.loc[mask]

def b_cut(df, val = 0):
    mask = (df.n_b > val) & (df.jet_b_index1 >= 0)
    return df.loc[mask]


#Cuts over the final Df

def pt_tau_cut(df, val = 25):
    # leading jets pt > 30 GeV
    mask = df.tau1_pT > val
    return df.loc[mask]

def pt_electron_cut(df, val = 25):
    # leading electron > 30 GeV
    mask = df.electron_pt > val
    return df.loc[mask]

def pt_muon_cut(df, val = 25):
    # leading muon > 30 GeV
    mask = df.muon_pt > val
    return df.loc[mask]

def pt_b_cut(df, val = 20):
    # leading jets pt > 30 GeV
    mask = df.b_pT > val
    return df.loc[mask]


def eta_tau_cut(df, val = 2.3):
    # leading jets eta < 2.3
    mask = np.abs(df.tau1_eta) < val
    return df.loc[mask]


def eta_b_cut(df, val = 2.5):
    # leading jets eta < 2.5
    mask = np.abs(df.b_eta) < val
    return df.loc[mask]


def phi_tau_cut(df, val= 2.0):
    # delta phi between met and tau > 2.0
    mask = np.abs(df.Delta_phi_Tau_Met) > val
    return df.loc[mask]


def et_met_cut(df, val = 200):
    # Met et greater than 200
    mask = df.met > val
    return df.loc[mask]


def final_cuts(df, pt_tau = 25, pt_b = 0, eta_tau = 2.3, eta_b = 2.5, met = 0, del_phi = 0):
    """
    Returns a copy of the df filtered by different variables and different objects
    Parameters:
        df : A Pandas.Dataframe to be filtered. This DataFrame must have a series of columns named as 
             'tau_pT' ,'tau_eta' ,'tau_phi' ,'tau_mass', 'b_pT' ,'b_eta' ,'b_phi' ,
             'b_mass','met_Met' ,'met_Phi' ,'met_Eta', 'n_tau', 'n_b'.
        pt_tau : Minimun value for tau's p_T.
        pt_b : Minimun value for b's p_T.
        eta_tau : Minimun value for tau's eta.
        eta_b : Minimun value for b's eta.
        met : Minimun value for p_T^{miss}. 
        del_phi : Minimun value for absolute value of delta phi between tau and met.
    """
    cut_df = df.copy()
    cut_df = pt_tau_cut(cut_df, pt_tau)
    cut_df = pt_b_cut(cut_df, pt_b)
    cut_df = eta_tau_cut(cut_df, eta_tau)
    cut_df = phi_tau_cut(cut_df, del_phi)
    cut_df = eta_b_cut(cut_df, eta_b)
    cut_df = et_met_cut(cut_df, met)
    return cut_df



def branch_index(df, branch):
    """
    Adds a column with the index of the first jet tagged as branch to the df.

    Branch:
    takes values of "jet_btag" or "jet_tautag"

    Ex:
    branch_index(cut_df, "jet_tautag")
    Returns a df with a column with the first jet tagged as tau per ivent
    """
    branch_jets = df[[f"{branch}{i}" for i in range(4)]].copy()

    # events with branch jets
    branch_events1 = branch_jets[branch_jets.sum(axis=1) > 0]
    
    # events with more tah 1 branch jets
    branch_events2 = branch_jets[branch_jets.sum(axis=1) > 1]

    # index of first branch jet
    branch_index1 = branch_events1.apply(lambda x: x > 0).apply(lambda x: np.nonzero(x.values)[0][0], axis=1)
    
    # index of first branch jet
    branch_index2 = branch_events2.apply(lambda x: x > 0).apply(lambda x: np.nonzero(x.values)[0][1], axis=1)

    # index of non branch jets (set to nan)
    branch_nan = pd.DataFrame(index= branch_jets[branch_jets.sum(axis=1) == 0].index)

    # branch jet index
    df["{}_index1".format(branch).replace('tag','')] = pd.concat([branch_index1,branch_nan]).sort_index()
    df["{}_index2".format(branch).replace('tag','')] = pd.concat([branch_index2,branch_nan]).sort_index()

    return df , branch_index1, branch_index2

def DeltaPhi(row, col1 = 'tau1_phi', col2 = 'met_Phi'):
    """
    correction on azimuthal angle difference dphi
    """
    dphi = row[col1] - row[col2]
    if dphi >= np.pi: 
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi

    return dphi


def DeltaPhi2(row, col1 = 'tau1_phi', col2 = 'met_Phi'):
    """
    correction on azimuthal angle difference dphi
    """
    dphi = row[col1] - row[col2]
    if dphi >= np.pi: 
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi

    return np.abs(dphi)


def m_Tot(row):  
    #Calculates TotalMass from 5.4  https://arxiv.org/pdf/1709.07242.pdf   
    pt1 = row['tau1_pT']
    px1 = row['tau1_pT'] * np.cos(row['tau1_phi'])
    py1 = row['tau1_pT'] * np.sin(row['tau1_phi'])
    pt2 = row['b_pT']
    px2 = row['b_pT'] * np.cos(row['b_phi'])
    py2 = row['b_pT'] * np.sin(row['b_phi'])
    met_pt3 = row['met_Met']
    px3 = met_pt3 * np.cos(row['met_Phi'])
    py3 = met_pt3 * np.sin(row['met_Phi'])


    vec1 = np.array([px1 , py1])
    vec2 = np.array([px2 , py2])
    vec3 = np.array([px3 , py3])
    vect_t = vec1 + vec2 + vec3 
    vec_t2 = np.dot(vect_t, vect_t)
    sum_escal = (pt1 + pt2 + met_pt3) **2
    return (sum_escal - vec_t2  ) ** 0.5


def transverse_mass(tau_pt, met_et, deltaphi):
    #Calculates the transverse mass between tau (or any other jet) and the met
    return np.sqrt(2 * tau_pt * met_et * (1 - np.cos(deltaphi)))


def invariant_mass(obj1_pt, obj1_eta, obj2_pt, obj2_eta, deltaphi ):
    #Calculates the invariant mass for 2 different objects
    return np.sqrt(2 * obj1_pt * obj2_pt * (np.cosh(obj1_eta-obj2_eta) - np.cos(deltaphi)))





def generate_ploteable_invariant_masses(df1, df2, df3):
    #Generates a series for the invariant masses between tau and MET for each dataframe  
    m1 = invariant_mass(df1.tau1_pT, df1.tau1_eta, df1.b_pT, df1.b_eta, df1.Delta_phi_Tau_B )
    m2 = invariant_mass(df2.tau1_pT, df2.tau1_eta, df2.b_pT, df2.b_eta, df2.Delta_phi_Tau_B )
    m3 = invariant_mass(df3.tau1_pT, df3.tau1_eta, df3.b_pT, df3.b_eta, df3.Delta_phi_Tau_B )
    return m1, m2, m3

def generate_ploteable_tr_tau_nu_masses(df1, df2, df3):
    #Generates a series for the transverse masses between tau and MET for each dataframe  
    m1 = transverse_mass(df1.tau1_pT, df1.met_Met, df1.Delta_phi_Tau_Met)
    m2 = transverse_mass(df2.tau1_pT, df2.met_Met, df2.Delta_phi_Tau_Met)
    m3 = transverse_mass(df3.tau1_pT, df3.met_Met, df3.Delta_phi_Tau_Met)
    return m1, m2, m3

def generate_ploteable_tr_b_nu_masses(df1, df2, df3):
    #Generates a series for the transverse masses between B and MET for each dataframe 
    m1 = transverse_mass(df1.b_pT, df1.met_Met, df1.Delta_phi_B_Met)
    m2 = transverse_mass(df2.b_pT, df2.met_Met, df2.Delta_phi_B_Met)
    m3 = transverse_mass(df3.b_pT, df3.met_Met, df3.Delta_phi_B_Met)
    return m1, m2, m3

def generate_ploteable_total_masses(df1, df2, df3):
    #Generates a series for the total msses between B , tau and MET for each dataframe  
    m1 = df1.apply(m_Tot, axis = 1)
    m2 = df2.apply(m_Tot, axis = 1)
    m3 = df3.apply(m_Tot, axis = 1)
    return m1, m2, m3


def Get_Pt_Eta_Phi_Tau_B(row):
    """
    Returns a row with a list of the Pt, Eta, Phi
    of the Taus and B's (in the respective order)
    Needs a column named Tau_b_Tuple.
    """
    if row['jet_tau_index1'] != row['jet_b_index1']:
        if pd.isna(row['jet_tau_index2']):

            n_tau1 = int(row['jet_tau_index1'])
            n_tau2 = np.NaN
            n_b1 = int(row['jet_b_index1'])
            num_b = row['n_b']

            s = pd.Series(cols)
            s1 = list(s[s.astype(str).str[-1] == str(n_tau1)][:4])
            s2 = list(s[s.astype(str).str[-1] == str(n_b1)][:4])


            tau_pT1 = row[str(s1[0])] 
            tau_eta1 = row[str(s1[1])] 
            tau_phi1 = row[str(s1[2])] 
            tau_mass1 = row[str(s1[3])]

            tau_pT2 = np.NaN
            tau_eta2 = np.NaN 
            tau_phi2 = np.NaN 
            tau_mass2 = np.NaN

            b_pT = row[str(s2[0])] 
            b_eta = row[str(s2[1])] 
            b_phi = row[str(s2[2])] 
            b_mass = row[str(s2[3])]

            met_Met = row['missinget_met']
            met_Phi = row['missinget_phi']
            met_Eta = row['missinget_eta']

        else:
            n_tau1 = int(row['jet_tau_index1'])
            n_tau2 = int(row['jet_tau_index2'])
            n_b1 = int(row['jet_b_index1'])
            num_b = row['n_b']

            s = pd.Series(cols)
            s1 = list(s[s.astype(str).str[-1] == str(n_tau1)][:4])
            s1_2 = list(s[s.astype(str).str[-1] == str(n_tau2)][:4])
            s2 = list(s[s.astype(str).str[-1] == str(n_b1)][:4])


            tau_pT1 = row[str(s1[0])] 
            tau_eta1 = row[str(s1[1])] 
            tau_phi1 = row[str(s1[2])] 
            tau_mass1 = row[str(s1[3])]

            tau_pT2 = row[str(s1_2[0])] 
            tau_eta2 = row[str(s1_2[1])] 
            tau_phi2 = row[str(s1_2[2])] 
            tau_mass2 = row[str(s1_2[3])]

            b_pT = row[str(s2[0])] 
            b_eta = row[str(s2[1])] 
            b_phi = row[str(s2[2])] 
            b_mass = row[str(s2[3])]

            met_Met = row['missinget_met']
            met_Phi = row['missinget_phi']
            met_Eta = row['missinget_eta']
    
    else:
        if not pd.isna(row["jet_b_index2"]):
            n_tau1 = int(row['jet_tau_index1'])
            n_tau2 = np.NaN
            n_b1 = int(row['jet_b_index2'])
            num_b = row['n_b']

            s = pd.Series(cols)
            s1 = list(s[s.astype(str).str[-1] == str(n_tau1)][:4])
            s2 = list(s[s.astype(str).str[-1] == str(n_b1)][:4])


            tau_pT1 = row[str(s1[0])] 
            tau_eta1 = row[str(s1[1])] 
            tau_phi1 = row[str(s1[2])] 
            tau_mass1 = row[str(s1[3])]

            tau_pT2 = np.NaN
            tau_eta2 = np.NaN 
            tau_phi2 = np.NaN 
            tau_mass2 = np.NaN

            b_pT = row[str(s2[0])] 
            b_eta = row[str(s2[1])] 
            b_phi = row[str(s2[2])] 
            b_mass = row[str(s2[3])]

            met_Met = row['missinget_met']
            met_Phi = row['missinget_phi']
            met_Eta = row['missinget_eta']
            
        elif not pd.isna(row["jet_tau_index2"]):
            n_tau1 = int(row['jet_tau_index2'])
            n_tau2 = np.NaN
            n_b1 = int(row['jet_b_index1'])
            num_b = row['n_b']

            s = pd.Series(cols)
            s1 = list(s[s.astype(str).str[-1] == str(n_tau1)][:4])
            s2 = list(s[s.astype(str).str[-1] == str(n_b1)][:4])


            tau_pT1 = row[str(s1[0])] 
            tau_eta1 = row[str(s1[1])] 
            tau_phi1 = row[str(s1[2])] 
            tau_mass1 = row[str(s1[3])]

            tau_pT2 = np.NaN
            tau_eta2 = np.NaN 
            tau_phi2 = np.NaN 
            tau_mass2 = np.NaN

            b_pT = row[str(s2[0])] 
            b_eta = row[str(s2[1])] 
            b_phi = row[str(s2[2])] 
            b_mass = row[str(s2[3])]

            met_Met = row['missinget_met']
            met_Phi = row['missinget_phi']
            met_Eta = row['missinget_eta']
        
        else:
            n_tau1 = np.NaN
            n_tau2 = np.NaN
            n_b1 = np.NaN
            num_b = np.NaN

            s = pd.Series(cols)
            s1 = list(s[s.astype(str).str[-1] == str(n_tau1)][:4])
            s2 = list(s[s.astype(str).str[-1] == str(n_b1)][:4])


            tau_pT1 = np.NaN 
            tau_eta1 = np.NaN
            tau_phi1 = np.NaN 
            tau_mass1 = np.NaN

            tau_pT2 = np.NaN
            tau_eta2 = np.NaN 
            tau_phi2 = np.NaN 
            tau_mass2 = np.NaN

            b_pT = np.NaN 
            b_eta = np.NaN
            b_phi = np.NaN 
            b_mass = np.NaN

            met_Met = np.NaN
            met_Phi = np.NaN
            met_Eta = np.NaN
            
            

    return (tau_pT1, tau_eta1, tau_phi1, tau_mass1, 
            tau_pT2, tau_eta2, tau_phi2, tau_mass2, 
            b_pT, b_eta, b_phi, b_mass, 
            met_Met, met_Phi, met_Eta, n_tau1, n_tau2, num_b, n_b1)

def generate_data_b_tau_nu(df):
    """Returns a Dataframe with the information per event
    of the tau_jet and the missin energy.
    The index preserves the index from the original dataframe.
    Arguments:
        df :  dataframe generated with 
        get_braches("MissingET.MET","MissingET.Eta","MissingET.Phi","Jet.PT",
                    "Jet.Eta","Jet.Phi","Jet.Mass","Jet.TauTag","Jet.BTag","Jet_size")

    Also the dataframe must content a column named as n_tau as the number of taus per event
    Columns : 
        'tau_pT' ,'tau_eta' ,'tau_phi' ,'tau_mass', 'b_pT' ,'b_eta' ,'b_phi' ,
        'b_mass','met_Met' ,'met_Phi' ,'met_Eta', 'n_tau', 'n_b'.

    """
    cut_df, tau_index1, tau_index2  = branch_index(df, 'jet_tautag')
    cut_df, b_index1, b_index2 = branch_index(cut_df, 'jet_btag')
    cut_df = tau_cut(cut_df)
    cut_df = b_cut(cut_df)

    s = cut_df.apply(Get_Pt_Eta_Phi_Tau_B, axis = 1)
    Df = pd.DataFrame(s.to_list(), index = s.index, 
                      columns=['tau1_pT' ,
                               'tau1_eta' ,
                               'tau1_phi' ,
                               'tau1_mass', 
                               'tau2_pT', 
                               'tau2_eta', 
                               'tau2_phi', 
                               'tau2_mass',
                               'b_pT' ,
                               'b_eta' ,
                               'b_phi' ,
                               'b_mass',
                               'met_Met' ,
                               'met_Phi' ,
                               'met_Eta' ,
                               'n_tau1', 
                               'n_tau2', 
                               'num_b','n_b1'])
  
    Df['Delta_phi_Tau_Met'] = Df.apply(DeltaPhi,axis = 1)
    Df['Delta_phi_B_Met'] = Df.apply(DeltaPhi,axis = 1, args=('b_phi', 'met_Phi'))
    Df['Delta_phi_Tau_B'] = Df.apply(DeltaPhi,axis = 1, args=('tau1_phi', 'b_phi'))
    return Df

def plot_significances(s, b1, b2, b3, var, w, rango, txt, labs_sizes = 20):
    """
    Returns a plot for the Significance Z between a signal and three backgrounds.
    Parameters:
        s : Signal, a Dataframe which is going to act as the numerator in the significance definition.
        b1 : Background number 1, a Dataframe which is going to act and
             as part of the denominator in the significance definition.
        b2 : Background number 2, a Dataframe which is going to act and
             as part of the denominator in the significance definition.
        b3 : Background number 3, a Dataframe which is going to act and
             as part of the denominator in the significance definition.
        var : The variable to be cut in order to maximize the significances. 
             Must be a function with the ending "_cut" implemented
             at the begginig of this script.
        w : A weights vector, a iterable ordered with the weight of the signal and the backgrounds.
        rango: The range of the variable to be filtered.        
    """
    arr_range = np.linspace(rango[0], rango[1], rango[2])

    sign1 = np.array([var(s, val = i).shape[0] * w[0] \
              / ((var(b1, val = i).shape[0]* w[1]) +
                 (var(b2, val = i).shape[0] * w[2]) +
                 (var(b3, val = i).shape[0] * w[3]) +
                var(s, val = i).shape[0] * w[0]) ** 0.5 \
             for i in arr_range])

    plt.figure(figsize=(8, 7))
    plt.plot(arr_range, sign1,'cs--')
    plt.ylabel('Significance', fontsize = labs_sizes)
    plt.xlabel(rf'${txt}$',fontsize = labs_sizes)
    #plt.title('CMS$\it{Simulation}$', loc='left', fontweight='bold')
    plt.title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    #plt.savefig(f'{Path_files}/Significance_{txt}')

def plot_mass1(m_type,s1,s2,s3,bg1,bg2,bg3, Ws, labels, bines = [30], labs_sizes = 23):
    """
    Returns a mass distributions for 3 signals and tree backgrounds
    Parameters:
        m_type: A string that will serve for the plot's x_label.
        s1 : Signal#1, a series or iterable to be binned.
        s2 : Signal#2, a series or iterable to be binned.
        s3 : Signal#3, a series or iterable to be binned.
        bg1 : Background#1, a series or iterable to be binned.
        bg2 : Background#2, a series or iterable to be binned.
        bg3 : Background#3, a series or iterable to be binned.
        Ws : The weights of the signal and background 
             in the order they are implemented in the function.
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        bines : bins for each histogram. Eachone must be a iterable of the edges of the bins
                ordered betweeen the minimun and the maximun value of the histograms
             
    """
    f, axs = plt.subplots( figsize=(9, 7))

    h1, binss1 = np.histogram(s1, bines)
    h2, binss2 = np.histogram(s2, bines)
    h3, binss3 = np.histogram(s3, bines)

    h4, binsb1 = np.histogram(bg1, bines)
    h5, binsb2 = np.histogram(bg2, bines)
    h6, binsb3 = np.histogram(bg3, bines)

    
    hep.histplot([h4 * Ws[3]/ bg1.shape[0], h5 * Ws[4] /bg2.shape[0], h6 * Ws[5] /bg3.shape],
                 bins = binss3, 
                 ax = axs,
                 color = ['c', 'm', 'y'],  
                 stack = True, 
                 histtype = 'fill', 
                 label = labels[3:],
                 #sort='label_l')
                 sort='yield')
    
    hep.histplot((h6 * Ws[5]/ bg3.shape[0]) + (h5 * Ws[4]/ bg2.shape[0]) + (h4 * Ws[3]/ bg1.shape[0]),
                 bins = binss3,
                 ax=axs, 
                 histtype='errorbar',
                 #hatch = '///',
                 yerr= (h6 ** 0.5 * Ws[5]/ bg3.shape[0]) + (h5** 0.5 * Ws[4]/ bg2.shape[0]) + (h4** 0.5 * Ws[3]/ bg1.shape[0]), 
                 c='black',
                 marker="",
                 capsize=4) 
                 #label = 'background err')
    
    hep.histplot(h1 * Ws[0]/ s1.shape[0],
                 bins = binss1,
                 ax = axs,
                 yerr = h1 ** 0.5 * Ws[0]/ s1.shape[0] ,
                 color = 'b',
                 histtype = 'step',  
                 label = labels[0])
    hep.histplot(h2 * Ws[1]/ s2.shape[0],
                 bins = binss1,
                 ax = axs,
                 yerr = h2 ** 0.5 * Ws[1]/ s2.shape[0],
                 color = 'r',
                 histtype = 'step',  
                 label = labels[1])
    hep.histplot(h3 * Ws[2]/ s3.shape[0],
                 bins = binss1,
                 ax = axs,
                 yerr = h3 ** 0.5 * Ws[2]/ s3.shape[0],
                 color = 'g',
                 histtype = 'step',  
                 label = labels[2])

    #hep.cms.label()
    axs.set_xlabel(m_type, fontsize = labs_sizes)
    axs.set_ylabel('Events',fontsize = labs_sizes)
    axs.set_yscale("log")
    axs.legend(fontsize = labs_sizes - 6 , loc = 'best')
    axs.xaxis.set_tick_params(labelsize= labs_sizes)
    axs.yaxis.set_tick_params(labelsize= labs_sizes)
    axs.set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    


def plot_Met_ET_Phi1(df1, df2, df3, bg1, bg2, bg3, labels, size, labs_sizes = 20):

    """
    Returns a 1 dimentional array with 6 histograms in 2 axis with the information of 
    missing energy in the transverse plane and the angle phi in that plane.
    Parameters:
        df1 : Signal#1, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        df2 : Signal#2, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        df3 : Signal#3, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg1 : Background#1, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg2 : Background#2, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg3 : Background#3, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi". 
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        size : Fountsize for the legend.
    """
    
    fig, axs = plt.subplots(1, 2, figsize = (14, 6), constrained_layout=True)

    bins1 = np.linspace(0,1500,30)
    s1_1, binss1 = np.histogram(df1.missinget_met, bins1)
    s2_1, binss2 = np.histogram(df2.missinget_met, bins1)
    s3_1, binss3 = np.histogram(df3.missinget_met, bins1)
    b1_1, binss1 = np.histogram(bg1.missinget_met, bins1)
    b2_1, binss2 = np.histogram(bg2.missinget_met, bins1)
    b3_1, binss3 = np.histogram(bg3.missinget_met, bins1)
    
    hep.histplot([b1_1, b2_1, b3_1],
                 bins = bins1,
                 stack = True,
                 density = True,
                 ax = axs[0],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_1, s2_1, s3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0],
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins2 = np.linspace(-np.pi,np.pi,25)
    s1_2, binss1 = np.histogram(df1.missinget_phi, bins2)
    s2_2, binss2 = np.histogram(df2.missinget_phi, bins2)
    s3_2, binss3 = np.histogram(df3.missinget_phi, bins2)
    b1_2, binss1 = np.histogram(bg1.missinget_phi, bins2)
    b2_2, binss2 = np.histogram(bg2.missinget_phi, bins2)
    b3_2, binss3 = np.histogram(bg3.missinget_phi, bins2)
    
    hep.histplot([b1_2, b2_2, b3_2],
                 bins = bins2,
                 density = True,
                 ax = axs[1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_r') 
    hep.histplot([s1_2, s2_2, s3_2],
                 bins = bins2,
                 ax = axs[1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    #hep.cms.label(ax=axs[0])
    #hep.cms.label(ax=axs[1])
    axs[0].legend(loc = 'best', fontsize = size)
    axs[1].legend(loc = 'lower center', fontsize = size)
    axs[0].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[0].set_xlabel(r"$p_T^m$", fontsize = labs_sizes)
    axs[1].set_ylabel("a.u.")
    axs[1].set_xlabel(r"$\phi(p_t^m)$")
    axs[0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    
    
def plot_Met_ET_Phi2(df1, df2, df3, bg1, bg2, bg3, labels, size, labs_sizes = 20):

    """
    Returns a 1 dimentional array with 6 histograms in 2 axis with the information of 
    missing energy in the transverse plane and the angle phi in that plane.
    Parameters:
        df1 : Signal#1, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        df2 : Signal#2, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        df3 : Signal#3, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg1 : Background#1, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg2 : Background#2, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi".
        bg3 : Background#3, a Pandas DataFrame with 2 columns named as "met_Met" and "met_Phi". 
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        size : Fountsize for the legend.
    """
    fig, axs = plt.subplots(1, 2, figsize = (14, 6), constrained_layout=True)

    bins1 = np.linspace(0,1300,50)
    s1_1, binss1 = np.histogram(df1.met_Met, bins1)
    s2_1, binss2 = np.histogram(df2.met_Met, bins1)
    s3_1, binss3 = np.histogram(df3.met_Met, bins1)
    b1_1, binss1 = np.histogram(bg1.met_Met, bins1)
    b2_1, binss2 = np.histogram(bg2.met_Met, bins1)
    b3_1, binss3 = np.histogram(bg3.met_Met, bins1)
    
    hep.histplot([b1_1, b2_1, b3_1],
                 bins = bins1,
                 density = True,
                 alpha = 0.7,
                 ax = axs[0],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_1, s2_1, s3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0],
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins2 = np.linspace(-np.pi,np.pi,25)
    s1_2, binss1 = np.histogram(df1.met_Phi, bins2)
    s2_2, binss2 = np.histogram(df2.met_Phi, bins2)
    s3_2, binss3 = np.histogram(df3.met_Phi, bins2)
    b1_2, binss1 = np.histogram(bg1.met_Phi, bins2)
    b2_2, binss2 = np.histogram(bg2.met_Phi, bins2)
    b3_2, binss3 = np.histogram(bg3.met_Phi, bins2)
    
    hep.histplot([b1_2, b2_2, b3_2],
                 bins = bins2,
                 density = True,
                 alpha = 0.7,
                 ax = axs[1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_2, s2_2, s3_2],
                 bins = bins2,
                 ax = axs[1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    #hep.cms.label(ax=axs[0])
    #hep.cms.label(ax=axs[1])
    axs[0].legend(loc = 'best', fontsize = size)
    axs[1].legend(loc = 'lower center', fontsize = size)
    axs[0].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[0].set_xlabel(r"$p_T^{miss}$", fontsize = labs_sizes)
    axs[1].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[1].set_xlabel(r"$\phi(p_T^{miss})$", fontsize = labs_sizes)
    axs[0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)

    
    
def plot_pt_eta_phi1(labels, df1, df2, df3, bg1, bg2, bg3, i, size, labs_sizes = 20):
    
    """
    Returns a 2 dimentional array with 6 histograms in 3 axis with the information of 
    pt eta and phi for a jet.
    Parameters:
        df1 : Signal#1, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        df2 : Signal#2, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        df3 : Signal#3, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        bg1 : Background#1, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        bg2 : Background#2, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        bg3 : Background#3, a Pandas DataFrame with 2 columns named as "jet_pt0", "jet_eta0" and "jet_phi0".
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        size : Fountsize for the legend.
        i : The i-th jet ordered in p_T from the hightest to the lowest
    """
    fig, axs = plt.subplots(2, 2, figsize = (14, 12), constrained_layout=True)
    fig.suptitle('Variables del jet_{}'.format(i), fontsize = 22)
    
    bins1 = np.linspace(0,1500,30)
    s1_1, binss1 = np.histogram(df1['jet_pt{}'.format(i)], bins1)
    s2_1, binss2 = np.histogram(df2['jet_pt{}'.format(i)], bins1)
    s3_1, binss3 = np.histogram(df3['jet_pt{}'.format(i)], bins1)
    b1_1, binss1 = np.histogram(bg1['jet_pt{}'.format(i)], bins1)
    b2_1, binss2 = np.histogram(bg2['jet_pt{}'.format(i)], bins1)
    b3_1, binss3 = np.histogram(bg3['jet_pt{}'.format(i)], bins1)
    
    hep.histplot([b1_1, b2_1, b3_1],
                 bins = bins1,
                 stack = True,
                 density = True,
                 ax = axs[0,0],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_1, s2_1, s3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0,0],
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins2 = np.linspace(-4.5,4.5,25)
    s1_2, binss1 = np.histogram(df1['jet_eta{}'.format(i)], bins2)
    s2_2, binss2 = np.histogram(df2['jet_eta{}'.format(i)], bins2)
    s3_2, binss3 = np.histogram(df3['jet_eta{}'.format(i)], bins2)
    b1_2, binss1 = np.histogram(bg1['jet_eta{}'.format(i)], bins2)
    b2_2, binss2 = np.histogram(bg2['jet_eta{}'.format(i)], bins2)
    b3_2, binss3 = np.histogram(bg3['jet_eta{}'.format(i)], bins2)
    
    hep.histplot([b1_2, b2_2, b3_2],
                 bins = bins2,
                 density = True,
                 ax = axs[0,1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_2, s2_2, s3_2],
                 bins = bins2,
                 ax = axs[0,1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins3 = np.linspace(-np.pi, np.pi,25)
    s1_3, binss1 = np.histogram(df1['jet_phi{}'.format(i)], bins3)
    s2_3, binss2 = np.histogram(df2['jet_phi{}'.format(i)], bins3)
    s3_3, binss3 = np.histogram(df3['jet_phi{}'.format(i)], bins3)
    b1_3, binss1 = np.histogram(bg1['jet_phi{}'.format(i)], bins3)
    b2_3, binss2 = np.histogram(bg2['jet_phi{}'.format(i)], bins3)
    b3_3, binss3 = np.histogram(bg3['jet_phi{}'.format(i)], bins3)
    
    hep.histplot([b1_3, b2_3, b3_3],
                 bins = bins3,
                 density = True,
                 ax = axs[1,0],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_3, s2_3, s3_3],
                 bins = bins3,
                 ax = axs[1,0],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins4 = np.linspace(0, np.pi,25)
    s1_4, binss1 = np.histogram(np.abs(df1.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    s2_4, binss2 = np.histogram(np.abs(df2.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    s3_4, binss3 = np.histogram(np.abs(df3.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    b1_4, binss1 = np.histogram(np.abs(bg1.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    b2_4, binss2 = np.histogram(np.abs(bg2.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    b3_4, binss3 = np.histogram(np.abs(bg3.apply(DeltaPhi, axis = 1, args= (f'jet_phi{i}', "missinget_phi"))), bins4)
    
    
    hep.histplot([b1_4, b2_4, b3_4],
                 bins = bins4,
                 density = True,
                 ax = axs[1,1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_4, s2_4, s3_4],
                 bins = bins4,
                 ax = axs[1,1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])
    
    #hep.cms.label(ax=axs[0,0])
    #hep.cms.label(ax=axs[0,1])
    #hep.cms.label(ax=axs[1,0])
    #hep.cms.label(ax=axs[1,1])
    axs[0,0].legend(loc = 'best', fontsize = size)
    axs[0,1].legend(loc = 'upper left', fontsize = size)
    axs[1,0].legend(loc = 'lower center', fontsize = size)
    axs[1,1].legend(loc = 'upper center', fontsize = size)
    
    axs[0,0].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[0,0].set_xlabel("$p_t(jet_{})$".format(i), fontsize = labs_sizes)
    axs[0,1].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[0,1].set_xlabel(r"$\eta (jet_{})$".format(i) , fontsize = labs_sizes)
    axs[1,0].set_ylabel("a.u.", fontsize = labs_sizes)
    axs[1,0].set_xlabel(r"$\phi (jet_{})$".format(i) , fontsize = labs_sizes)
    axs[1,1].set_ylabel("a.u.")
    axs[1,1].set_xlabel(r"$\Delta\phi (jet_{}, p_T^m)$".format(i))
    axs[0,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[0,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)


def plot_pt_eta_phi2(labels, obj, df1, df2, df3, bg1, bg2, bg3, size = 14, labs_sizes = 20):  
    
    """
    Returns a 2 dimentional array with 6 histograms in 3 axis with the information of 
    pt eta and phi for a jet.
    Parameters:
        obj : The object type, must be a string, can be "tau" or "b"
        df1 : Signal#1, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        df2 : Signal#2, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        df3 : Signal#3, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg1 : Background#1, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg2 : Background#2, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg3 : Background#3, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        size : Fountsize for the legend.
    """
    fig, axs = plt.subplots(2 , 2 ,figsize=(18,14), constrained_layout=True)
    bins1 = np.linspace(0,1300,50)
    s1_1, binss1 = np.histogram(df1[f"{obj}_pT"], bins1)
    s2_1, binss2 = np.histogram(df2[f"{obj}_pT"], bins1)
    s3_1, binss3 = np.histogram(df3[f"{obj}_pT"], bins1)
    b1_1, binss1 = np.histogram(bg1[f"{obj}_pT"], bins1)
    b2_1, binss2 = np.histogram(bg2[f"{obj}_pT"], bins1)
    b3_1, binss3 = np.histogram(bg3[f"{obj}_pT"], bins1)
    
    hep.histplot([b1_1, b2_1, b3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0,0],
                 color = ['c', 'm', 'y'],
                 alpha = 0.7,
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_1, s2_1, s3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0,0],
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins2 = np.linspace(-4.5,4.5,25)
    s1_2, binss1 = np.histogram(df1[f"{obj}_eta"], bins2)
    s2_2, binss2 = np.histogram(df2[f"{obj}_eta"], bins2)
    s3_2, binss3 = np.histogram(df3[f"{obj}_eta"], bins2)
    b1_2, binss1 = np.histogram(bg1[f"{obj}_eta"], bins2)
    b2_2, binss2 = np.histogram(bg2[f"{obj}_eta"], bins2)
    b3_2, binss3 = np.histogram(bg3[f"{obj}_eta"], bins2)
    
    hep.histplot([b1_2, b2_2, b3_2],
                 bins = bins2,
                 alpha = 0.7,
                 density = True,
                 ax = axs[0,1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_2, s2_2, s3_2],
                 bins = bins2,
                 ax = axs[0,1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins3 = np.linspace(-np.pi, np.pi,25)
    s1_3, binss1 = np.histogram(df1[f"{obj}_phi"], bins3)
    s2_3, binss2 = np.histogram(df2[f"{obj}_phi"], bins3)
    s3_3, binss3 = np.histogram(df3[f"{obj}_phi"], bins3)
    b1_3, binss1 = np.histogram(bg1[f"{obj}_phi"], bins3)
    b2_3, binss2 = np.histogram(bg2[f"{obj}_phi"], bins3)
    b3_3, binss3 = np.histogram(bg3[f"{obj}_phi"], bins3)
    
    hep.histplot([b1_3, b2_3, b3_3],
                 bins = bins3,
                 density = True,
                 alpha = 0.7,
                 ax = axs[1,0],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_3, s2_3, s3_3],
                 bins = bins3,
                 ax = axs[1,0],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])

    
    bins4 = np.linspace(0, np.pi,25)
    s1_4, binss1 = np.histogram(np.abs(df1.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    s2_4, binss2 = np.histogram(np.abs(df2.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    s3_4, binss3 = np.histogram(np.abs(df3.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    b1_4, binss1 = np.histogram(np.abs(bg1.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    b2_4, binss2 = np.histogram(np.abs(bg2.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    b3_4, binss3 = np.histogram(np.abs(bg3.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    
    
    hep.histplot([b1_4, b2_4, b3_4],
                 bins = bins4,
                 density = True,
                 alpha = 0.7,
                 ax = axs[1,1],
                 color = ['c', 'm', 'y'],
                 histtype = 'fill', 
                 label = labels[3:],
                 sort= 'label_l') 
    hep.histplot([s1_4, s2_4, s3_4],
                 bins = bins4,
                 ax = axs[1,1],
                 density = True,
                 histtype='step',
                 color = ['b', 'r', 'g'],
                 label = labels[:3])
    
    #hep.cms.label(ax=axs[0,0])
    #hep.cms.label(ax=axs[0,1])
    #hep.cms.label(ax=axs[1,0])
    #hep.cms.label(ax=axs[1,1])
    axs[0,0].legend(loc = 'best', fontsize = size)
    axs[0,1].legend(loc = 'upper left', fontsize = size)
    axs[1,0].legend(loc = 'lower center', fontsize = size)
    axs[1,1].legend(loc = 'upper center', fontsize = size)

    if obj == 'tau1':
        axs[0,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,0].set_xlabel(f"$p_T(\{obj[:-1]})$", fontsize = labs_sizes)
        axs[0,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,1].set_xlabel(rf"$\eta (\{obj[:-1]})$", fontsize = labs_sizes)
        axs[1,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,0].set_xlabel(rf"$\phi (\{obj[:-1]})$", fontsize = labs_sizes)
        axs[1,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,1].set_xlabel(rf"$\Delta\phi (\{obj[:-1]}, p_T^m)$", fontsize = labs_sizes)

    else:
        axs[0,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,0].set_xlabel("$p_T({})$".format(obj), fontsize = labs_sizes)
        axs[0,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,1].set_xlabel(r"$\eta ({})$".format(obj), fontsize = labs_sizes)
        axs[1,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,0].set_xlabel(r"$\phi ({})$".format(obj), fontsize = labs_sizes)
        axs[1,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,1].set_xlabel(r"$\Delta\phi ({}, p_T^m)$".format(obj), fontsize = labs_sizes)
                
    axs[0,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[0,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    
    
def plot_pt_eta_phi2_2(labels, obj, bg2, bg3, size = 14, labs_sizes = 20):  
    
    """
    Returns a 2 dimentional array with 6 histograms in 3 axis with the information of 
    pt eta and phi for a jet.
    Parameters:
        obj : The object type, must be a string, can be "tau" or "b"
        df1 : Signal#1, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        df2 : Signal#2, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        df3 : Signal#3, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg1 : Background#1, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg2 : Background#2, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        bg3 : Background#3, a Pandas DataFrame with 2 columns named as "obj_pT", "obj_eta" and "obj_phi".
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        size : Fountsize for the legend.
    """
    fig, axs = plt.subplots(2 , 2 ,figsize=(18,14), constrained_layout=True)
    bins1 = np.linspace(0,1300,50)
    b2_1, binss2 = np.histogram(bg2[f"{obj}_pT"], bins1)
    b3_1, binss3 = np.histogram(bg3[f"{obj}_pT"], bins1)
    
    hep.histplot([b2_1, b3_1],
                 bins = bins1,
                 density = True,
                 ax = axs[0,0],
                 color = ['m', 'y'],
                 alpha = 0.7,
                 histtype = 'fill', 
                 label = labels[4:],
                 sort= 'label_l') 

    
    bins2 = np.linspace(-4.5,4.5,25)
    #b1_2, binss1 = np.histogram(bg1[f"{obj}_eta"], bins2)
    b2_2, binss2 = np.histogram(bg2[f"{obj}_eta"], bins2)
    b3_2, binss3 = np.histogram(bg3[f"{obj}_eta"], bins2)
    
    hep.histplot([b2_2, b3_2],
                 bins = bins2,
                 alpha = 0.7,
                 density = True,
                 ax = axs[0,1],
                 color = ['m', 'y'],
                 histtype = 'fill', 
                 label = labels[4:],
                 sort= 'label_l') 

    
    bins3 = np.linspace(-np.pi, np.pi,25)
    #b1_3, binss1 = np.histogram(bg1[f"{obj}_phi"], bins3)
    b2_3, binss2 = np.histogram(bg2[f"{obj}_phi"], bins3)
    b3_3, binss3 = np.histogram(bg3[f"{obj}_phi"], bins3)
    
    hep.histplot([b2_3, b3_3],
                 bins = bins3,
                 density = True,
                 alpha = 0.7,
                 ax = axs[1,0],
                 color = ['m', 'y'],
                 histtype = 'fill', 
                 label = labels[4:],
                 sort= 'label_l') 


    
    bins4 = np.linspace(0, np.pi,25)
    #b1_4, binss1 = np.histogram(np.abs(bg1.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    b2_4, binss2 = np.histogram(np.abs(bg2.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    b3_4, binss3 = np.histogram(np.abs(bg3.apply(DeltaPhi, axis = 1, args= ('{}_phi'.format(obj), "met_Phi"))), bins4)
    
    
    hep.histplot([b2_4, b3_4],
                 bins = bins4,
                 density = True,
                 alpha = 0.7,
                 ax = axs[1,1],
                 color = ['m', 'y'],
                 histtype = 'fill', 
                 label = labels[4:],
                 sort= 'label_l') 
    
    #hep.cms.label(ax=axs[0,0])
    #hep.cms.label(ax=axs[0,1])
    #hep.cms.label(ax=axs[1,0])
    #hep.cms.label(ax=axs[1,1])
    axs[0,0].legend(loc = 'best', fontsize = size)
    axs[0,1].legend(loc = 'upper left', fontsize = size)
    axs[1,0].legend(loc = 'lower center', fontsize = size)
    axs[1,1].legend(loc = 'upper center', fontsize = size)
    
    axs[0,0].set(xlim=(-50, 800))
    axs[0,1].set(xlim=(-3, 3))

    if obj == 'tau1':
        axs[0,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,0].set_xlabel(f"$p_T(\{obj[:-1]})$", fontsize = labs_sizes)
        axs[0,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,1].set_xlabel(rf"$\eta (\{obj[:-1]})$", fontsize = labs_sizes)
        axs[1,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,0].set_xlabel(rf"$\phi (\{obj[:-1]})$", fontsize = labs_sizes)
        axs[1,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,1].set_xlabel(rf"$\Delta\phi (\{obj[:-1]}, p_T^m)$", fontsize = labs_sizes)

    else:
        axs[0,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,0].set_xlabel("$p_T({})$".format(obj), fontsize = labs_sizes)
        axs[0,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[0,1].set_xlabel(r"$\eta ({})$".format(obj), fontsize = labs_sizes)
        axs[1,0].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,0].set_xlabel(r"$\phi ({})$".format(obj), fontsize = labs_sizes)
        axs[1,1].set_ylabel("a.u.", fontsize = labs_sizes)
        axs[1,1].set_xlabel(r"$\Delta\phi ({}, p_T^m)$".format(obj), fontsize = labs_sizes)
                
    axs[0,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[0,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,0].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    axs[1,1].set_title(r'$150 \,fb^{-1}(13 \,TeV)$', loc = 'right', fontsize = labs_sizes + 1)
    
    
    
def Binner(m_type,s1,s2,s3,bg1,bg2,bg3, Ws, labels, bines = [30], labs_sizes = 20):
    """
    Returns a mass distributions values for 3 signals and 3 backgrounds
    Parameters:
        m_type: A string that will serve for the plot's x_label.
        s1 : Signal#1, a series or iterable to be binned.
        s2 : Signal#2, a series or iterable to be binned.
        s3 : Signal#3, a series or iterable to be binned.
        bg1 : Background#1, a series or iterable to be binned.
        bg2 : Background#2, a series or iterable to be binned.
        bg3 : Background#3, a series or iterable to be binned.
        Ws : The weights of the signal and background 
             in the order they are implemented in the function.
        labels : Labels to recognize the histograms. Must be a iterable of strings.
        bines : bins for each histogram. Eachone must be a iterable of the edges of the bins
                ordered betweeen the minimun and the maximun value of the histograms
             
    """

    h1, binss1 = np.histogram(s1, bines)
    h2, binss2 = np.histogram(s2, bines)
    h3, binss3 = np.histogram(s3, bines)

    h4, binsb1 = np.histogram(bg1, bines)
    h5, binsb2 = np.histogram(bg2, bines)
    h6, binsb3 = np.histogram(bg3, bines)

    df = pd.DataFrame([h1 * Ws[0]/ s1.shape[0],
                       h2 * Ws[1]/ s2.shape[0],
                       h3 * Ws[2]/ s3.shape[0],
                       h4 * Ws[3]/ bg1.shape[0], 
                       h5 * Ws[4] /bg2.shape[0], 
                       h6 * Ws[5] /bg3.shape[0]],
                     index = labels)
    df_err = pd.DataFrame([h1 ** 0.5 * Ws[0]/ s1.shape[0],
                          h2 ** 0.5 * Ws[1]/ s2.shape[0],
                          h3 ** 0.5 * Ws[2]/ s3.shape[0],
                          h4 ** 0.5 * Ws[3]/ bg1.shape[0], 
                          h5 ** 0.5 * Ws[4] /bg2.shape[0], 
                          h6 ** 0.5 * Ws[5] /bg3.shape[0]],
                          index = labels)
    return df, binss1, df_err

def get_max(significance_dictionary):
    maximum = 0
    for variable, significance in significance_dictionary.items():
        if significance > maximum:
            maximum = significance
            var_max = variable
    return var_max


def optimization(signal_data, backgrounds_data, weights, variable, variable_range, signal_name):
    """
    
    """
    rango = np.linspace(variable_range[0], variable_range[1], variable_range[2])
    events_signal = np.array([signal_data[signal_data[variable] > i].shape[0] for i in rango]) * weights[signal_name]
    #events_background = np.array([np.sum(background_data[background_data[variable] > i]['weigth']) for i in rango]) * weights[1]
    events_background = sum(np.array([backgrounds_data[background].query(f"{variable} > {i}").shape[0] \
                                      for i in rango]) * weights[background] for background in backgrounds_data)
        
    
    
    sig = events_signal / (events_signal + events_background) ** 0.5
    sig = sig / np.max(sig)
    significance = {rango[i]: sig[i] for i in range(len(sig))}
    return significance