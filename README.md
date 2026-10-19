
## Experimental Results

### Shot-Scale Classification

Two CNN experiments were evaluated on the same held-out test set of 500 images, with 100 images per class.

| Metric         |     Baseline CNN | Augmented CNN |
| -------------- | ---------------: | ------------: |
| Test accuracy  | **57.40%** |        51.60% |
| Macro F1-score | **57.63%** |        47.88% |
| Best epoch     |               10 |             5 |
| Training time  |    12.18 minutes |  5.64 minutes |

The baseline CNN achieved better test accuracy and macro F1-score than the augmented CNN. The augmentation experiment used horizontal flipping and small brightness and contrast variations, but these transformations did not improve the recorded test performance.

The baseline model is therefore the preferred model among these two experiments based on the saved evaluation results.

### Camera-Movement Classification

Both camera-movement experiments were evaluated on 1,682 test samples.

| Metric            |     Standard CNN | Balanced-Sampling CNN |
| ----------------- | ---------------: | --------------------: |
| Test accuracy     | **78.89%** |                27.23% |
| Macro F1-score    | **35.18%** |                18.16% |
| Weighted F1-score | **78.87%** |                37.84% |
| Test loss         |            0.842 |                 1.690 |

The standard model achieved substantially better overall results. However, the dataset is strongly imbalanced, and its high accuracy does not indicate equally strong performance across all classes.

The standard model achieved F1-scores of 86.37% for `Static` and 71.80% for `Motion`, but 17.74% for `Push` and 0% for both `Pull` and `Multi_movement`.

The balanced-sampling experiment improved recall for `Push` and `Pull`, but reduced overall accuracy and macro F1-score. Neither experiment correctly classified any `Multi_movement` test samples.

### Limitations

- The camera-movement test set contains 1,120 `Static` samples but only 8 `Multi_movement` samples.
- Overall accuracy can hide weak performance on minority classes.
- The balanced-sampling experiment did not provide a better overall result in the recorded comparison.
- The camera-movement labels are the original MovieShots labels. They are not mapped to specific movements such as pan, tilt, zoom, or tracking/dolly.
- The recorded results should be interpreted using both aggregate metrics and per-class evaluation.

### Results Interpretation

The baseline shot-scale CNN is the stronger of the two recorded shot-scale experiments. The standard camera-movement CNN is the stronger overall camera-movement experiment, although further work is needed to improve minority-class performance.
