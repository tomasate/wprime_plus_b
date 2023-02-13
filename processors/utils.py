import os
import json
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Union
from coffea.nanoevents.methods import candidate, vector

loc_base = os.environ["PWD"]
def normalize(var: ak.Array, cut: ak.Array = None) -> ak.Array:
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
        ar = ak.to_numpy(ak.fill_none(var[cut], np.nan))
        return ar


def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True,
) -> Union[ak.Array, np.ndarray]:
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(
            ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None
        )
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret


def build_p4(cand: ak.Array) -> ak.Array:
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


def save_dfs_parquet(fname: str, dfs_dict: dict) -> None:
    """
    save dataframes as parquet files
    """
    table = pa.Table.from_pandas(dfs_dict)
    if len(table) != 0:  # skip dataframes with empty entries
        pq.write_table(table, fname + ".parquet")


def ak_to_pandas(output_collection: dict) -> pd.DataFrame:
    """
    cast awkward array into a pandas dataframe
    """
    output = pd.DataFrame()
    for field in ak.fields(output_collection):
        output[field] = ak.to_numpy(output_collection[field])
    return output


def save_output(
    events: ak.Array,
    dataset: str,
    output: pd.DataFrame,
    year: str,
    channel: str,
    output_location: str,
    dir_name: str,
) -> None:
    """
    creates output folders and save dfs to parquet files
    """
    with open(f"{loc_base}/data/simplified_samples.json", "r") as f:
        simplified_samples = json.load(f)
    sample = simplified_samples[year][dataset]
    partition_key = events.behavior["__events_factory__"]._partition_key.replace(
        "/", "_"
    )
    date = datetime.today().strftime("%Y-%m-%d")

    # creating directories for each channel and sample
    if not os.path.exists(
        output_location + date + "/" + dir_name + "/" + year + "/" + channel
    ):
        os.makedirs(
            output_location + date + "/" + dir_name + "/" + year + "/" + channel
        )
    if not os.path.exists(
        output_location
        + date
        + "/"
        + dir_name
        + "/"
        + year
        + "/"
        + channel
        + "/"
        + sample
    ):
        os.makedirs(
            output_location
            + date
            + "/"
            + dir_name
            + "/"
            + year
            + "/"
            + channel
            + "/"
            + sample
        )
    fname = (
        output_location
        + date
        + "/"
        + dir_name
        + "/"
        + year
        + "/"
        + channel
        + "/"
        + sample
        + "/"
        + partition_key
    )
    save_dfs_parquet(fname, output)