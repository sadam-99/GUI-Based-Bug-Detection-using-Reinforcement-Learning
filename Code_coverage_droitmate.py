# Code(Script) for the calculation of code coverage %
# Shivam Gupta

import numpy as np
import nltk
import math
import re
import os
import sys
import glob 

coverage_dir = r"F:\University of Texas at Dallas(UTD) MS-CS Fall 2019\Research_UTD_Wei_Yang_Lab\droidMate\droidMate\coverage"
original_path= r"F:\University of Texas at Dallas(UTD) MS-CS Fall 2019\Research_UTD_Wei_Yang_Lab\droidMate\droidMate\com.xlythe.calculator.material_93.apk.json"

code_cover_lines = []
orig_code_lines = []
# Lists of all file paths
files_list = os.listdir(coverage_dir)
for files in files_list:
    coverage_file =os.path.join(coverage_dir,files) 
    with open(coverage_file, encoding="utf8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for t in lines:
            tw = t.split(';')[0]
            code_cover_lines.append(tw)

with open(original_path, encoding="utf8") as fi:
  lin = fi.readlines()
  lin = [y.strip() for y in lin[3:-2]]
  for c in lin:
    cc = c.split(':')[0]
    cc = re.sub('"|"', '', cc)
    orig_code_lines.append(cc)

code_cover_lines = np.unique(code_cover_lines)
orig_code_lines = np.unique(orig_code_lines)

    
p = set(code_cover_lines)&set(orig_code_lines)
print('code lines covered :', len(code_cover_lines))
print('code lines in the original apk :', len(orig_code_lines))
# Percentage of Code coverage
perc_code_coverage = len(p)/len(orig_code_lines)*100
print('The code covearage % is', perc_code_coverage)


