# PathSnap - Model Evaluation Results

## Overview
The trained final_model was tested using the FastAPI prediction endpoint
with HMM-based route prediction enabled.

## Evaluation Results

| Trip ID | GPS Points | Top-1 Accuracy | Arc Precision | Arc Recall | Route Continuity | Inference Time |
|---------|------------|----------------|---------------|------------|------------------|----------------|
| 00000015 | 6854 | 57.43% | 40.56% | 96.32% | 92.63% | 1.47 s |
| 00000067 | 916 | 88.65% | 79.17% | 92.68% | 92.55% | 0.30 s |
| 00000094 | 3230 | 85.48% | 81.14% | 91.89% | 84.86% | 1.02 s |

## Observations
- All three trips produced successful API predictions.
- Trip 00000067 achieved the highest point-level accuracy.
- Trip 00000094 achieved the highest arc precision.
- Trip 00000015 needs further improvement in route precision.
- HMM-based prediction improved route continuity on trip 00000015
  compared with classifier-only prediction.

## Conclusion
The prediction pipeline successfully processes GPS trips and produces
road-network route predictions. Further tuning is required to improve
precision and directional consistency across challenging trips.
