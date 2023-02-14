#!/bin/bash

echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`" 

export X509_USER_PROXY=$HOME/tmp/x509up
cd /afs/cern.ch/user/t/tatehort/wprime_plus_b/

python3 run.py --processor ttbar --executor futures --workers 4 --sample WJetsToLNu_HT-1200To2500 --channel mu --nfiles 1 --year 2017