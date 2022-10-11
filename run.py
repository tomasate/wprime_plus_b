import os
import sys
import json
import pickle
import argparse
import dask
from datetime import datetime
from coffea import processor
from dask.distributed import Client, PipInstall
from typing import List


def main(args):
    loc_base = os.environ['PWD']

    # define processor
    if args.processor == "ttbar":
        from analysis.ttbar_processor import TTBarControlRegionProcessor
        proc = TTBarControlRegionProcessor
                
            
    # executor arguments
    if args.executor == "iterative":
        
        executor_args = {
            "schema": processor.NanoAODSchema,
        }
        
    if args.executor == "dask":
        client = Client(
            "tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786"
        )
        # https://github.com/dask/distributed/issues/6202
        plugin = PipInstall(packages=["git+https://github.com/deoache/b_lepton_met.git"])
        client.register_worker_plugin(plugin)
        
        executor_args = {"schema": processor.NanoAODSchema, "client": client}

        
    # load fileset
    with open(
        f"{loc_base}/data/fileset/fileset_{args.year}_UL_NANO.json", "r"
    ) as f:
        fileset = json.load(f)

    for key, val in fileset.items():
        if val is not None:
            if args.nfiles == -1:
                fileset[key] = ["root://xcache/" + file for file in val]
            else:
                fileset[key] = ["root://xcache/" + file for file in val[: args.nfiles]]
                
                
    # run processor
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=proc(
            year=args.year,
            yearmod=args.yearmod,
            output_location=args.output_location,
            dir_name=args.dir_name,
        ),
        executor=(
            processor.iterative_executor
            if args.executor == "iterative"
            else processor.dask_executor
        ),
        executor_args=executor_args,
    )

    # save dictionary with cutflows
    date = datetime.today().strftime("%Y-%m-%d")
    with open(
        args.output_location + args.dir_name + date + "/" + "out.pkl", "wb"
    ) as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="ttbar",
        help="processor to run",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="iterative",
        help="executor (iterative or dask)",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=1,
        help="number of files per sample",
    )
    parser.add_argument("--year", dest="year", type=str, default="2017", help="year")
    parser.add_argument(
        "--yearmod", dest="yearmod", type=str, default="", help="year modifier"
    )
    parser.add_argument(
        "--output_location",
        dest="output_location",
        type=str,
        default="/home/cms-jovyan/b_lepton_met/data/outfiles/",
        help="output location",
    )
    parser.add_argument(
        "--dir_name",
        dest="dir_name",
        type=str,
        default="control_region/",
        help="output directory name",
    )

    args = parser.parse_args()
    main(args)
