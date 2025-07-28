# Module 0
### Deadline: 23/03
- everybody experiments

everybody looks for which outlier detection they would rather perform

## Meeting 2 objectives
create standardized dataset to work on
handle features accordingly to updated description.txt
outlier detection


> # !!!!!TODO first!!!!!
> decide which type of outlier detection each of use prefers
> __Done__




# Module 1
## Anomaly detection (and visualization w/ dim redu)
### Deadline: 30/03
splits decided prev week


### TODO
finish with feats handling (missing handling, representations etc)
have some base for outlier det
start w/ imbalanced learning (just some groundwork)
    which model?


how should we represent genres?
- word2vec (solve problem of nans)
- linear comb of one-hot enc
- one-hot enc (bruno's pick for ML models)


## Imbalanced Learning
### Deadline: 06/04
__tosplit__


### meeting 11/04
some comparisons with outlier det methods


## goals for 18th
- having a finalized for comparison representation for outlier detection
- statistical imputation of outliers for all feats
  - **DONE**
- knn+DTs for imbalanced learning (both undersampling and oversampling)
  - chiara covered smote, adasyn, decision threshold, and the combination
  - ankit covers class weights, decision threshold
  - bruno covers undersampling, mix between under/over
    - random, knn based done
    - TODO: medoid/centroid based


## goals for 29th, 14:00
- think of classif target
  - maybe start doing something?
- get t-sne to work on more dimensions
  - actually doesn't make too much sense; rather, used pairplot to underline the distribution on 2 dims
- re tune imbalanced learning for newly defined labels [bruno] 
  - DONE
- test with class weights for imbalanced [bruno]
  - DONE
- in general, be done with module 1


# Classification
target: titletype

# Regression
target: avgrating


# Tasks
## Classification
- nns [bruno]
- logistic reg [chiara]
- svm [chiara]
- ensemble [bruno]
- GBM [ankit]
## Regression
- model 1 [chiara]
- model 2 [bruno]
## Explainability
- exp [ankit]

# Module 3
- motifs and discords [ankit]
- clustering 1 [chiara]
- clustering 2 [ankit]
- knn [chiara]
- second classif method [bruno]