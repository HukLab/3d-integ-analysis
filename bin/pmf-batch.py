import os.path
import subprocess
basedir = os.path.abspath('../plots')

def run(cmd, elbs, outs):
    for i, outdir in zip(elbs, outs):
        outdir = os.path.join(basedir, outdir)
        print i, outdir
        subprocess.call(cmd.format(i, outdir), shell=True)

# elbs = [1, 2]
# outs = ['twin-limb-free', 'tri-limb-free']
# cmd = "python pmf_fit.py -l -e {0} -n 20 -b 1000 --outdir '{1}'"
# run(cmd, elbs, outs)

# elbs = [1, 2]
# outs = ['twin-limb-zero', 'tri-limb-zero']
# cmd = "python pmf_fit.py -l -e {0} -n 20 -b 1000 --enforce-zero --outdir '{1}'"
# run(cmd, elbs, outs)

elbs = [1]
outs = ['twin-limb-zero-drop_two']
cmd = "python pmf_fit.py -l -e {0} -n 20 -b 1000 --enforce-zero --min-di 3 --outdir '{1}'"
run(cmd, elbs, outs)

elbs = [1]
outs = ['twin-limb-drop_two']
cmd = "python pmf_fit.py -l -e {0} -n 20 -b 1000 --min-di 3 --outdir '{1}'"
run(cmd, elbs, outs)
