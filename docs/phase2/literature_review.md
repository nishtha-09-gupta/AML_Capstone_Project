# Literature review and architecture rationale

1. **Pearl (1988), Probabilistic Reasoning in Intelligent Systems**, Morgan Kaufmann. The project-supplied foundational reference motivates representing conditional independence explicitly. Our disease-rooted graph is a generative factorization, not a claim that disease causes age or sex. Its omitted dependencies and the oldpeak–slope edge are provisional assumptions. Existing edge evidence and limitations are in `docs/bn_design.md` and `config/bn_graph.json`.
2. **UCI Heart Disease documentation**, [dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), [DOI](https://doi.org/10.24432/C52P4X). Defines recorded disease categories 0–4 and the conventional presence/absence task. The supplied merged CSV has 920 rows, so it must not be described as the original 303-record processed Cleveland file. Our secondary Cleveland analysis uses the merged CSV slice of D, with all historical T rows excluded.
3. **Gorishniy et al. (2021), Revisiting Deep Learning Models for Tabular Data**, [paper](https://arxiv.org/abs/2106.11959). Studies strong neural baselines and consistent benchmarking; finds no universally superior learner. This supports a modest, explicit MLP baseline and matching folds. Our two-layer network is not their ResNet or FT-Transformer, and our small study cannot reproduce their benchmark conclusions.
4. **Grinsztajn, Oyallon and Varoquaux (2022), Why do tree-based models still outperform deep learning on typical tabular data?**, [paper](https://arxiv.org/abs/2207.08815). Demonstrates competitive tree baselines and differing inductive biases on tabular benchmarks. We retain LR and RF controls rather than assuming DL must improve results. Their benchmarks are not evidence of a clinical benefit on this dataset.
5. **Srivastava et al. (2014), Dropout: A Simple Way to Prevent Neural Networks from Overfitting**, [JMLR](https://www.jmlr.org/papers/v15/srivastava14a.html). Motivates dropout to reduce co-adaptation. We test dropout 0.2 against zero dropout; usefulness is measured rather than presumed.
6. **Guo et al. (2017), On Calibration of Modern Neural Networks**, [PMLR](https://proceedings.mlr.press/v70/guo17a.html). Distinguishes accurate classification from reliable confidence and studies post-processing calibration. We measure Brier score, log loss and five-bin reliability. We do not claim that the fitted MLP is clinically calibrated, and do not fit a calibrator on historical T.

## Why this architecture

The input is a small, unordered feature vector, without images or temporal sequences.
An MLP fits the data structure; CNNs and recurrent models would add unjustified
structure. ReLU hidden layers (32,16) permit continuous nonlinear interactions,
whereas the BN bins numeric values and assumes most conditional independences.
The sigmoid output estimates disease-presence probability under the sampled data.
Dropout, AdamW weight decay and training-only epoch selection constrain capacity.

## Hybrid contribution and limits

The BN-assisted MLP appends a cross-fitted standardized posterior log-odds feature.
It tests whether a coarse, structured probabilistic summary complements continuous
features. This is an application of feature stacking, not a new algorithm. It adds
32 first-layer weights and one BN fit at inference; it does not merely average
scores. Out-of-fold training scores prevent direct target leakage. The final BN
is refitted on all permitted training rows, creating an OOF-to-inference sample-size
shift that remains a limitation. The NB-assisted ablation tests whether the extra
oldpeak–slope dependence helps beyond a generic class-conditional summary.

The original feature-hybrid experiment does not show a hybrid advantage over the plain MLP.
A justified implementation and a reproducible negative result satisfy technical
investigation; they do not establish the successful added value needed to promise
hybrid bonus marks. Evaluation data were not used to search for a more flattering
variant.

## Follow-up calibration and fusion

[scikit-learn's StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
describes training the final estimator on cross-validated base predictions.
The follow-up applies this principle to two logits with penalized logistic fusion
and calibrated single-base ablations. Unlike fitting a meta-model on an existing
global OOF matrix, fresh cross-fitting is repeated within every nested training
subset. Guo et al. motivate checking confidence separately from ranking; the
present study uses logistic calibration and does not reproduce their temperature
scaling experiments. Residual correction is an alternative tested mechanism,
not a claim to implement a ResNet or reproduce image-recognition results.

## Verified medical hybrid and healthcare context

- **Zhao, Feng, Chen, Zhou and Yu (2022)**, *Diagnose Like a Radiologist: Hybrid Neuro-Probabilistic Reasoning for Attribute-Based Medical Image Diagnosis*. Verified author-posted preprint: [arXiv:2208.09282](https://arxiv.org/abs/2208.09282), DOI 10.48550/arXiv.2208.09282. Combines BN and graph-convolutional reasoning branches on medical images. It motivates complementary structured/neural evidence; this project's tabular logistic stack does not implement their attention mechanism or inherit their results.
- **Weng, Reps, Kai, Garibaldi and Qureshi (2017)**, *Can machine-learning improve cardiovascular risk prediction using routine clinical data?* PLoS ONE 12(4):e0174944. [Primary publication](https://doi.org/10.1371/journal.pone.0174944). Evaluates neural and other learners for ten-year cardiovascular events. Relevant cardiovascular neural context, with a different cohort and outcome from our recorded disease-presence task.
- **van der Laan, Polley and Hubbard (2007)**, *Super learner*. Statistical Applications in Genetics and Molecular Biology 6:Article25. [DOI](https://doi.org/10.2202/1544-6115.1309), [PubMed metadata](https://pubmed.ncbi.nlm.nih.gov/17910531/). Supports cross-validation-based prediction fusion. Our penalized two-logit stack is a narrower method, not a replication of all Super Learner theory.
- **Van Calster, McLernon, van Smeden, Wynants and Steyerberg (2019)**, *Calibration: the Achilles heel of predictive analytics*. BMC Medicine 17:230. [Publisher](https://doi.org/10.1186/s12916-019-1466-7). Connects calibration to medical prediction quality. Our modest Brier gain and sparse reliability bins do not establish clinical calibration or utility.

The Grinsztajn title now matches the [published NeurIPS 2022 title](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html), which includes *typical*. Citation verification concerns publication details and applicability, not independent replication of each paper's experimental findings.

See `results_interpretation.md` for OOF-to-full-fit distribution shift and post-selection uncertainty. These limitations apply to the calibrated stack as well as the original feature hybrid.
