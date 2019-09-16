from kachergis_model import subject_log_likelihood
import json
import sys
import datetime
import time
from scipy.optimize import minimize

# Example:
# python3 fit_llh_kach_scp.py o-1subj oscar_1subject_uniform > output-llh-kach-o-1subj.txt

data_short = sys.argv[1] 
data_long = sys.argv[2]

print("start time: {}".format(datetime.datetime.now()))
start = time.time()

outfilename = "results-llh-kach-"+data_short+".txt"
outfile = open(outfilename, 'w')
outfile.write("ID\t condition\t subj acc\t ML\t parameters\t\n")
outfile.close()

bnds = ((0.1, 1), (0.01, 10), (0.1, 30))

with open("data/"+data_long+"_preprocessed.json") as data_file:
    experiment = json.load(data_file)  
participants = len(experiment)

condition = ''
for subject in experiment:
    res = minimize(subject_log_likelihood, [0.97, 1, 3], args=(subject), bounds=bnds)
    subj_accuracy = round(subject['accuracy_score'][-1], 6)
    print(res['x'])
    outfile = open(outfilename, 'a')
    outfile.write("{}\t {}\t {}\t {}\t {}\t\n".format(subject["ID"], subject['condition'], subj_accuracy, round(res['fun'], 2), res['x']))
    outfile.close()

print("end time: {}".format(datetime.datetime.now()))
time_elapsed = time.time() - start
print("fitting took {} seconds per participant".format(time_elapsed / participants))
