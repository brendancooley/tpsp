from paramiko import SSHClient
from scp import SCPClient
import sys
import os
import imp
import getpass

import c_setup as setup
imp.reload(setup)

mil_off = sys.argv[1] == "True"

size = "mid/"
M = 100

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(hostname='adroit.princeton.edu', username="bcooley", password=getpass.getpass('password: '))

scp = SCPClient(ssh.get_transport())
sftp = ssh.open_sftp()

i = 1

for i in range(1, M+1):
    setup_hpc = setup.setup("hpc", size, bootstrap=True, bootstrap_id=i, mil_off=mil_off, base_path="/home/bcooley")
    setup_local = setup.setup("local", size, bootstrap=True, bootstrap_id=i, mil_off=mil_off)
    try:
        sftp.stat(setup_hpc.xlhvt_star_path)
        print("bootstrap id " + str(i) + " exists on hpc")
        sys.stdout.flush()
        if not os.path.exists(setup_local.xlhvt_star_path):
            print("transferring down bootstrap id " + str(i))
            scp.get(setup_hpc.xlhvt_star_path, setup_local.xlhvt_star_path)
    except IOError:
        if os.path.exists(setup_local.xlhvt_star_path):
            scp.put(setup_local.xlhvt_star_path, setup_hpc.xlhvt_star_path)
            print("transferring local bootstrap id " + str(i) + " to hpc...")
            sys.stdout.flush()
