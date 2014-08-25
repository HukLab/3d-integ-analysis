import glob
import json
import os.path

for infile in glob.glob('*.json'):
    obj = json.load(infile)
    
