import os
import json
import argparse
import os
from math import ceil
import logging
logging.basicConfig(filename="/afs/cern.ch/user/t/tatehort/wprime_plus_b/log.txt", level=logging.DEBUG)




with open('/afs/cern.ch/user/t/tatehort/wprime_plus_b/data/simplified_samples.json', 'r') as f:
  data = json.load(f)

samples = list(data["2017"].values())
for value in samples:

    submit= f"""executable            = /afs/cern.ch/user/t/tatehort/wprime_plus_b/submitters/process{value}.sh
arguments             = $(ClusterId)$(ProcId)
output                = output_logs/proc_t.$(ClusterId).$(ProcId).out
error                 = output_logs/proc_t.$(ClusterId).$(ProcId).err
log                   = output_logs/proc_t.$(ClusterId).log

+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
queue 1
    """
    with open(f"/afs/cern.ch/user/t/tatehort/wprime_plus_b/submitters/submit{value.replace('-', '_')}.sub", "w") as submit_writen:
        submit_writen.write(submit)

    arch_sh = f"""#!/bin/bash

echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`" 

export X509_USER_PROXY=$HOME/tmp/x509up
cd /afs/cern.ch/user/t/tatehort/wprime_plus_b/

python3 run.py --processor ttbar --executor futures --workers 4 --sample {value} --channel mu --nfiles 1 --year 2017"""

    with open(f"/afs/cern.ch/user/t/tatehort/wprime_plus_b/submitters/process{value}.sh", "w") as submit_writen:
        submit_writen.write(arch_sh)

paths = os.popen('find /afs/cern.ch/user/t/tatehort/wprime_plus_b/submitters/ -name "*.sub"').read().split()


for path in paths:
    os.system(f"condor_submit {path}")
    #print(path)

