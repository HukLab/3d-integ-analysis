import subprocess
from subprocess import Popen

for i in xrange(10, 30):
    # proc = Popen('python pmf_fit.py --plot -l -e 0 -n 20 -b 50 --outdir ../plots/pmf-boots/'.format(i))
    print 'NBINS = {0}'.format(i)
    proc = subprocess.call('python pmf_fit.py -l -e 0 -n {0} -b 0 --outdir ../plots/pmf-boots/'.format(i), shell=True)

"""
2d:
    * first slope: x<15, 15<x<17, 17<x
    * noise at end: high 20s, especially 21 and 25-28
3d:
    * not that affected when x>15
"""
