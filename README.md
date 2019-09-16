# xsl_word_learning

## Models implemented for: 

Khoe, Y. H., Perfors, A., & Hendrickson, A. T. (2019). Modeling individual performance in cross-situational word learning.  In A.K. Goel, C.M. Seifert, & C. Freksa (Eds.), Proceedings of the 41st Annual Conference of the Cognitive Science Society (pp. 560-566). Montreal, QB: Cognitive Science Society.

# Models
## Associative Uncertainty- & Familiarity-Biased Model

Kachergis, G., Yu, C., & Shiffrin, R. M. (2012, November). Cross-situational 
word learning is better modeled by associations than hypotheses. In Development 
and Learning and Epigenetic Robotics (ICDL), 2012 IEEE International Conference 
on (pp. 1-6). IEEE.

Files:
- Model: kachergis_model.py
- Fitting script: fit_llh_kach_scp.py

Fit this model to the data using:
```
python3 fit_llh_kach_scp.py o-1subj oscar_1subject_uniform > output-llh-kach-o-1subj.txt
```

## Propose but Verify

Trueswell, J. C., Medina, T. N., Hafri, A., & Gleitman, L. R. (2013). Propose but verify: 
Fast mapping meets cross-situational word learning. Cognitive psychology, 66(1), 126-156.

Files:
- Model: propose_but_verify_model.py
- Fitting script: fit_pbv_median_likelihood.py

Fit this model to the data using:
```
python3 fit_pbv_median_likelihood.py o-1subj oscar_1subject_uniform > output-mllh-pbv-o-1subj.txt
```

# Data

The experimental data from one participant is included in: 
- data/oscar_1subject_uniform_preprocessed.json
