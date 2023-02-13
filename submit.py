import os
import json
import argparse
import os
from math import ceil
import logging


loc_base = os.environ["PWD"]
logging.basicConfig(filename=f"{loc_base}/log.txt", level=logging.DEBUG)

try:
    os.mkdir(f"{loc_base}/submitters")
except Exception as e:
    pass

try:
    os.mkdir(f"{loc_base}/output_logs")
except Exception as e:
    pass
try:
    os.mkdir(f"{loc_base}/output")
except Exception as e:
    pass

with open(f'{loc_base}/data/simplified_samples.json', 'r') as f:
  data = json.load(f)

samples = list(data["2017"].values())
for value in samples:

    submit= f"""executable            = {loc_base}/submitters/process{value}.sh
arguments             = $(ClusterId)$(ProcId)
output                = output_logs/proc_t.$(ClusterId).$(ProcId).out
error                 = output_logs/proc_t.$(ClusterId).$(ProcId).err
log                   = output_logs/proc_t.$(ClusterId).log

+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
queue 1
    """
    with open(f"{loc_base}/submitters/submit{value.replace('-', '_')}.sub", "w") as submit_writen:
        submit_writen.write(submit)

    arch_sh = f"""#!/bin/bash

echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`" 

export X509_USER_PROXY=$HOME/tmp/x509up
cd {loc_base}/

python3 run.py --processor ttbar --executor futures --workers 4 --sample {value} --channel mu --nfiles 1 --year 2017"""

    with open(f"{loc_base}/submitters/process{value}.sh", "w") as submit_writen:
        submit_writen.write(arch_sh)

paths = os.popen(f'find {loc_base}/submitters/ -name "*.sub"').read().split()


for path in paths:
    #os.system(f"condor_submit {path}")
    print(path)

