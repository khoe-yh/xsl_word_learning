from propose_but_verify_model import subject_median_log_likelihood
import json
import sys
import datetime
import time
from pyswarm import pso

data_short = sys.argv[1]
data_long = sys.argv[2]

# Example:
# python3 fit_pbv_median_likelihood.py o-1subj oscar_1subject_uniform > output-mllh-pbv-o-1subj.txt

outfilename = "results-mllh-pbv-"+data_short+".txt"
data_file_path = "data/"+data_long+"_preprocessed.json"

print("start time: {}".format(datetime.datetime.now()))
start = time.time()

outfile = open(outfilename, 'w')
outfile.write("ID\t condition\t subj acc\t ML\t parameters\t\n")
outfile.close()

#   alpha   alpha_confirmed
lb = [0, 0]
ub = [1, 1]
bnds = ((0, 0.99), (0, 0.99))

maxiter = 50
swarmsize = 10
minfunc = 0

print("lb = {}".format(lb))
print("ub = {}".format(ub))
print("maxiter = {}".format(maxiter))
print("swarmsize = {}".format(swarmsize))
print("minfunc = {}".format(minfunc))


with open(data_file_path) as data_file:
    experiment = json.load(data_file)  
participants = len(experiment)

condition = ''
for subject in experiment:

    subj_accuracy = round(subject['accuracy_score'][-1], 6)
    
    # subject_median_log_likelihood will return the median of 10 runs by default
    # for 20 runs, change args to [subject, 20]
    args =  [subject]
    xopt, fopt = pso(subject_median_log_likelihood, lb, ub, args=args, minfunc=minfunc, swarmsize=swarmsize, debug=True, maxiter=maxiter)
    print(xopt)
    outfile = open(outfilename, 'a')
    outfile.write("{}\t {}\t {}\t {}\t {}\t\n".format(subject["ID"], subject['condition'], subj_accuracy, round(fopt, 2), xopt))
    
    outfile.close()

print("end time: {}".format(datetime.datetime.now()))
time_elapsed = time.time() - start
participant_seconds = time_elapsed / participants
print("fitting took {} seconds (or {} hours) per participant".format(participant_seconds, round(participant_seconds / 3600, 1)))
