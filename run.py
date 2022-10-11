import os
import sys
import json
import pickle
import argparse
import dask
from datetime import datetime
from coffea import processor
from dask.distributed import Client
from pathlib import Path
from distributed.diagnostics.plugin import UploadDirectory

def main(args):
    loc_base = os.environ['PWD']

    # load fileset
    with open(
        f"{loc_base}/fileset/fileset_{args.year}_UL_NANO.json", "r"
    ) as f:
        fileset = json.load(f)

    for key, val in fileset.items():
        if val is not None:
            if args.nfiles == -1:
                fileset[key] = ["root://xcache/" + file for file in val]
            else:
                fileset[key] = ["root://xcache/" + file for file in val[: args.nfiles]]

    # define processor
    if args.processor == "ttbar":
        from b_lepton_met.ttbar_cr_processor import TTBarControlRegionProcessor
        p = TTBarControlRegionProcessor
                
    # executor arguments
    if args.executor == "iterative":
        executor_args = {
            "schema": processor.NanoAODSchema,
        }
    elif args.executor == "dask":
        client = Client(
            "tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786"
        )
        def set_env():
            os.environ["PYTHONPATH"] = loc_base
        
        print(client.run(set_env))
        print(client.run(lambda: os.environ["PYTHONPATH"]))
        
        #print(dask.config.get("jobqueue.coffea-casa.local-directory"))
        #dask.config.set({"jobqueue.coffea-casa.local-directory": f"{loc_base}/analysis"})
        #os.system(F"export DASK_JOBQUEUE__COFFEA-CASA__LOCAL-DIRECTORY={loc_base}")
        #print(dask.config.get("jobqueue.coffea-casa.local-directory"))
        #print(json.dumps(dask.config.config, indent=4))
        # https://distributed.dask.org/en/stable/plugins.html#built-in-nanny-plugins

        try:
            client.register_worker_plugin(UploadDirectory(loc_base, restart=True), nanny=True)
        except OSError:
            print(f"failed to upload directory {loc_base}")
        """
        p = Path(f"{loc_base}/analysis").glob('**/*.py')
        files = [x for x in p if x.is_file()]
        for file in files:
            print(str(file), type(str(file)))
            client.upload_file(str(file))
        """
        executor_args = {"schema": processor.NanoAODSchema, "client": client}

    # run processor
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=p(
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
        default="/home/cms-jovyan/b_lepton_met/analysis/outfiles/",
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
