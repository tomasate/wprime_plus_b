import os
import json
import argparse
import os
from datetime import datetime
from math import ceil
import logging

processor = "ttbar"
year = "2017"
channel = "mu"

date = datetime.today().strftime("%Y-%m-%d")
loc_base = os.environ["PWD"]
logging.basicConfig(filename=f"{loc_base}/log.txt", level=logging.DEBUG)

try:
    os.mkdir(f"{loc_base}/submitters")
except Exception as e:
    pass

try:
    os.mkdir(f"{loc_base.replace('/wprime_plus_b', '')}/wprime_plus_b_logs")
except Exception as e:
    pass
try:
    os.mkdir(f"{loc_base}/output")
except Exception as e:
    pass

with open(f'{loc_base}/data/simplified_samples.json', 'r') as f:
  data = json.load(f)


samples = list(data[year].values())
for value in samples:


    submit= f"""executable            = {loc_base}/submitters/process{value}.sh
arguments             = $(ClusterId)$(ProcId)
output                = {loc_base.replace('/wprime_plus_b', '')}/wprime_plus_b_logs/proc_{value.replace('-', '_')}.$(ClusterId).$(ProcId).out
error                 = {loc_base.replace('/wprime_plus_b', '')}/wprime_plus_b_logs/proc_{value.replace('-', '_')}.$(ClusterId).$(ProcId).err
log                   = {loc_base.replace('/wprime_plus_b', '')}/wprime_plus_b_logs/proc_{value.replace('-', '_')}.$(ClusterId).log

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

python3 run.py --processor {processor} --executor futures --workers 4 --sample {value} --channel {channel} --nfiles -1 --year {year}

#move output to eos
#xrdcp -r -f output/ /eos/home-t/tatehort/
#rm {loc_base}/output/{date}/{processor}/{year}/{channel}/{value}.pkl"""


    with open(f"{loc_base}/submitters/process{value}.sh", "w") as submit_writen:
        submit_writen.write(arch_sh)

paths = os.popen(f'find {loc_base}/submitters/ -name "*.sub"').read().split()


#for path in paths:
#    os.system(f"condor_submit {path}")
    #print(path)

