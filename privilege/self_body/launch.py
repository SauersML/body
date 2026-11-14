import subprocess, sys
subprocess.Popen(["$BODY_WORK/islands/shell/hypha-rp-run", "dev", "hypha-prod-selfbody", "A100 SXM,A100 PCIe,H100 PCIe,H100 SXM,RTX PRO 6000", "5"],
                 stdout=open("launch.log", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=dict(__import__("os").environ, RP_PARALLEL="1"))
