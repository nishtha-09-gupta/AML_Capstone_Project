# Viva guide

**What is predicted?** Recorded disease presence: num > 0. The model does not predict future cardiovascular events or recommend treatment.

**Why feed-forward layers?** Each record is an unordered, small tabular vector. Two ReLU layers can represent nonlinear interactions without inventing image or sequence structure.

**What is the BN joint distribution?** P(Y) times the product of P(X_i|Y), replacing the slope term with P(slope|Y,oldpeak). Marginalize unavailable evidence. The disease-root directions express statistical factorization, not causation.

**How are numeric bins learned?** Median impute on permitted training rows, compute tertiles, remove duplicate and extreme cut points. Store exact cut points per fitted model. They are statistical bins, not medically established thresholds.

**What is Naive Bayes here?** The same discrete encoding and per-cell alpha=1 Dirichlet smoothing with the oldpeak-to-slope edge removed. This isolates graph structure from discretization differences.

**What is the hybrid?** Standardize logit(BN probability) from cross-fitted predictions and append it to ordinary MLP inputs. It offers a class-conditional summary while retaining continuous numeric inputs.

**Why cross-fitting?** In-sample BN predictions depend on their own labels through the fitted CPTs. Three-fold cross-fitting produces each training input using a BN fitted without that row. Even early-stop optimization rows use cross-fitting within that subset alone.

**How is early stopping isolated?** Split only the permitted fitting subset into optimization and early-stop validation rows. Fit its preprocessor and BN on optimization rows. Choose epochs by validation BCE; then reinitialize and refit on all permitted training rows for the chosen epoch count.

**Why three seeds?** They expose initialization and minibatch variability. Every seed is retained; seed 42 was declared as the primary saved-model comparison, not chosen after results.

**How was the primary family selected?** Average five-fold S F1 at 0.5 across declared seeds; AUC breaks ties. C selects each frozen model's threshold. T never selects families, seeds, epochs or cutoffs.

**What is BCE?** The mean of -[y log(p)+(1-y) log(1-p)]. BCEWithLogitsLoss computes it stably before applying a sigmoid for inference. AdamW provides decoupled weight decay; dropout masks hidden units during training only.

**What do ROC-AUC and PR-AUC mean?** ROC-AUC summarizes ranking across false-positive rates; PR-AUC summarizes precision/recall and depends on class prevalence. We report trapezoidal PR-AUC and average precision separately because they are different integrations.

**What is calibration?** Agreement between predicted probabilities and observed frequencies. Reliability bins show this descriptively; Brier is mean squared probability error. Good AUC alone does not imply calibration. Five bins and few records do not establish clinical calibration.

**Did the hybrid help?** No consistent benefit over the standalone MLP was demonstrated. Read the actual results and paired intervals. Do not say that a working combination proves innovation or earns bonus marks automatically.

**What do the ablations establish?** They isolate the BN input, dropout, weight decay and BN versus NB summary under fixed settings. They are empirical comparisons, not proofs; their results must be read alongside seed variability.

**What is the biggest limitation?** A small historical, merged dataset with an already-inspected holdout, uncertain patient linkage and hospital distribution shifts. Exact duplicate removal cannot prove independent patients. No prospective utility or causal claim follows.

**What remains human work?** Confirm author names and contributions, understand the code, rehearse the presentation, answer the viva independently, and obtain faculty acceptance of the provisional graph and dataset scope.
