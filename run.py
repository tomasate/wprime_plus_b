import os
import sys
import json
import pickle
import argparse
import dask
import importlib.resources
from datetime import datetime
from coffea import processor
from dask.distributed import Client
from distributed.diagnostics.plugin import UploadDirectory

def main(args):
    loc_base = os.environ["PWD"]

    # load fileset
    with open(f"{loc_base}/data/simplified_samples.json", "r") as f:
        simplified_samples = json.load(f)[args.year]
        simplified_samples = {v: k for k, v in simplified_samples.items()}
    
    with open(f"{loc_base}/data/fileset/fileset_{args.year}_UL_NANO.json", "r") as f:
        data = json.load(f)
        
    for key, val in data.items():
        if simplified_samples[args.sample] in key: 
            fileset = {key: val}
            if val is not None:
                if args.nfiles == -1:
                    fileset[key] = ["root://xcache/" + file for file in val]
                else:
                    fileset[key] = ["root://xcache/" + file for file in val[: args.nfiles]]

    # define processor
    if args.processor == "ttbar":
        from processors.ttbar_processor import TTBarControlRegionProcessor
        proc = TTBarControlRegionProcessor
        
    if args.processor == "trigger":
        from processors.trigger_efficiency_processor import TriggerEfficiencyProcessor
        proc = TriggerEfficiencyProcessor
        
    # executors and arguments
    executors = {
        "iterative": processor.iterative_executor,
        "futures": processor.futures_executor,
        "dask": processor.dask_executor,
    }
    executor_args = {
        "schema": processor.NanoAODSchema,
    }
    
    if args.executor == "futures":
        executor_args.update({"workers": args.workers})
        
    if args.executor == "dask":
        client = Client(
            "tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786"
        )
        try:
            client.register_worker_plugin(
                UploadDirectory(
                    f"{loc_base}/wprime_plus_b", restart=True, update_path=True
                ),
                nanny=True,
            )
        except OSError:
            print("Failed to upload the directory")
        executor_args.update({"client": client})
        
    # run processor
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=proc(
            year=args.year,
            yearmod=args.yearmod,
            channel=args.channel,
            output_location=args.output_location,
            dir_name=args.processor,
        ),
        executor=executors[args.executor],
        executor_args=executor_args,
    )

    # save dictionary with cutflows
    date = datetime.today().strftime("%Y-%m-%d")
    if not os.path.exists(
        args.output_location + date + "/" + args.processor + "/" + args.year + "/" + args.channel
    ):
        os.makedirs(
            args.output_location + date + "/" + args.processor + "/" + args.year + "/" + args.channel
        )
    with open(
        args.output_location
        + date
        + "/"
        + args.processor
        + "/"
        + args.year
        + "/"
        + args.channel
        + "/"
        + f"{args.sample}_out.pkl",
        "wb",
    ) as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        dest="sample",
        type=str,
        default="TTTo2L2Nu",
        help="sample to process",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="number of workers (futures executor)",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="ele",
        help="lepton channel {ele, mu}",
    )
    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="ttbar",
        help="processor to run {trigger, ttbar}",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="iterative",
        help="executor {iterative, futures, dask}",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=1,
        help="number of files per sample (all files: -1)",
    )
    parser.add_argument("--year", dest="year", type=str, default="2017", help="year")
    parser.add_argument(
        "--yearmod", dest="yearmod", type=str, default="", help="year modifier"
    )
    parser.add_argument(
        "--output_location",
        dest="output_location",
        type=str,
        default="/home/cms-jovyan/wprime_plus_b/outfiles/",
        help="output location",
    )

    args = parser.parse_args()
    main(args)
