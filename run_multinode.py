"""Wrap script for multi-node accellerate

Accelerate's multi-node support is based on the assumption
that all the nodes are known in advance and can be configured
to use the correct config file.  This does not match SLURM on
a shared GPU cluster's assumptions.

As a result we need to pass the rank through the command-line,
but the rank isn't known until the script has been invoked.

As a result, we need to bounce through this script to insert
the correct machine rank.

Note: This may be better written as a bash script.
"""


import os
import subprocess
import socket


nodelist = os.environ.get(
    "SLURM_JOB_NODELIST",
    socket.gethostname().split('.')[0],
)
print("Node List:", nodelist)

proc_id = os.environ.get("SLURM_PROCID", "0")
print("Rank:", proc_id)

output = subprocess.run(
    ["scontrol", "show", "hostnames", nodelist],
    capture_output=True,
)
full_nodelist = output.stdout.decode('utf8').split()
print("Full nodelist:", full_nodelist)

num_machines = len(full_nodelist)
print("num_machines", num_machines)

main_node = full_nodelist[0]
print("Main node:", main_node)

print(f"Running {proc_id}")
proc = subprocess.Popen(
    [
        "accelerate",
        "launch",
        "--multi-gpu",
        "--num_processes",
        "2",
        "--num_machines",
        str(num_machines),
        "--main_process_ip",
        main_node,
        "--main_process_port",
        "29500",
        "--machine_rank",
        proc_id,
        "llm_fine_tuning.py",
        "--num-epochs",
        "2",
    ],
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

print("Streaming stdout...")
for line in proc.stdout:
    print(line)

proc.poll()
print(proc.returncode)
