This is the closest reproduction. 

Metrics:

==================================================
DPO PREDICTIONS EVALUATION RESULTS
==================================================

Total Samples: 1452

--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
                 Predicted
                 0      1
Actual    0      793    126   (TN=793, FP=126)
          1      177    356   (FN=177, TP=356)

--------------------------------------------------
CLASSIFICATION METRICS
--------------------------------------------------
Accuracy:     0.7913  (79.13%)
Precision:    0.7386  (Class 1: Polarized)
Recall:       0.6679  (Class 1: Polarized)
F1-Score:     0.7015  (Class 1: Polarized)

--------------------------------------------------
DETAILED CLASSIFICATION REPORT
--------------------------------------------------

Per-Class Metrics:
Class 0 (Not Polarized):
  Precision: 0.8175
  Recall:    0.8629
  F1-Score:  0.8396
  Support:   919

Class 1 (Polarized):
  Precision: 0.7386
  Recall:    0.6679
  F1-Score:  0.7015
  Support:   533

Macro Average:
  Precision: 0.7781
  Recall:    0.7654
  F1-Score:  0.7705

==================================================

==================================================
SKLEARN CLASSIFICATION REPORT
==================================================
                   precision    recall  f1-score   support

Not Polarized (0)       0.82      0.86      0.84       919
    Polarized (1)       0.74      0.67      0.70       533

         accuracy                           0.79      1452
        macro avg       0.78      0.77      0.77      1452
     weighted avg       0.79      0.79      0.79      1452