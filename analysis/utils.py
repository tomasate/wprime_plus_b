import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate, vector

def normalize(var: ak.Array, cut: ak.Array = None):
    """
    normalize arrays after a cut or selection
    
    params:
    -------
    var:
        variable array
    cut:
        mask array to filter variable array
    """
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(var, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar
    
def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret

def build_p4(cand: ak.Array):
    """
    builds a 4-vector
    
    params:
    -------
    cand:
        candidate array
    """
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )