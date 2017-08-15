
# run_argo.py
import subprocess
# Define command and arguments
command = 'Rscript'
path2script = 'run_argo.R'
# Variable number of args in a list
args = ['../../data/evaluation_config/mansa_config_01.csv']
# Build subprocess command
cmd = [command, path2script] + args
# check_output will run the command and store to result
x = subprocess.check_output(cmd, universal_newlines=True)
print('config file is', x)
