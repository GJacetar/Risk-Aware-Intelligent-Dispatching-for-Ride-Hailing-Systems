# CC-STMT Prediction Module Description

The `CC-STMT` directory contains the prediction component of the proposed framework. CC-STMT stands for Closed-Loop Context-aware Spatio-Temporal Multi-Task Network. It is designed to generate dispatch-oriented probabilistic prediction fields for ride-hailing systems.

The prediction module is organized as follows:

```text
CC-STMT/
├── common/
├── NYC/
├── Chicago-A/
└── Chicago-B/
```

The `common` folder contains the shared CC-STMT model definition. The city folders contain city-specific scripts for data processing, model training, and prediction evaluation.

## Module Purpose

The purpose of the CC-STMT module is to estimate future regional states that can be used by the dispatching module. Unlike ordinary demand forecasting models that only predict future order counts, CC-STMT produces multiple prediction outputs:

```text
latent demand intensity
low-opportunity risk
demand dispersion
auxiliary speed prediction
```

These outputs are used not only for prediction evaluation but also for downstream dispatching. In particular, the demand intensity and low-opportunity risk fields are transferred to the DP-OTM module to construct destination opportunity and destination risk terms.

## Shared Model Code

The shared model code is placed in:

```text
CC-STMT/common/
```

This folder contains the model architecture used by all city settings. The model is city-independent because it operates on processed tensors and graph matrices with a unified format.

The model expects the following inputs:

```text
historical demand sequence
traffic-speed sequence
en-route supply feedback sequence
weather or contextual feature sequence
physical-distance adjacency matrix
functional-semantic adjacency matrix
```

The main model components include:

```text
demand-gated speed denoising
multi-source feature fusion
dual-graph constrained attention encoding
historical-window decoding
ZINB-based probabilistic output heads
auxiliary speed prediction head
```

The physical-distance graph captures spatial proximity and road-network related relationships. The functional-semantic graph captures similarity in regional functions or historical mobility patterns.

## City-Specific Folders

The prediction module includes three city-specific folders:

```text
CC-STMT/NYC/
CC-STMT/Chicago-A/
CC-STMT/Chicago-B/
```

Each city folder contains the scripts needed to prepare and evaluate CC-STMT for that specific experimental setting. The city-level scripts are separated because the three settings use different data sources, spatial units, temporal resolutions, and preprocessing rules.

A typical city folder contains:

```text
data_processor.py
train.py
evaluate.py
```

The exact content may vary slightly depending on the city setting.

## Data Processing

The data processing script converts raw public trip records and spatial information into the tensor format used by CC-STMT.

The expected outputs include:

```text
demand_tensor_full.npy
speed_tensor_full.npy
enroute_tensor_full.npy
weather_tensor.npy
adj_spatial.npy
adj_semantic.npy
dist_matrix.npy
zone_mapping.csv
```

The demand tensor records regional request counts over time. The speed tensor represents regional traffic-speed states derived from completed trips. The en-route tensor records vehicles that are currently serving passengers and are expected to re-enter the system in future time slots. The weather tensor provides contextual external features. The graph matrices define the spatial and semantic relationships between regions.

The generated tensors follow the general shape:

```text
time slots × spatial zones × feature dimensions
```

The graph matrices follow the shape:

```text
spatial zones × spatial zones
```

## Training Script

The training script estimates the CC-STMT parameters using the processed tensors. It follows a chronological split of the time series. The earlier part of the data is used for training, the middle part for validation, and the final part for testing.

The training process includes:

```text
loading processed tensors
normalizing input features using training-period statistics
constructing historical-window samples
loading graph matrices
training CC-STMT
selecting the best checkpoint according to validation performance
saving the trained model checkpoint
```

The main model checkpoint is typically saved as:

```text
best_CC_STMT_model.pth
```

The training script is intended to document the training logic of the proposed model. Since this repository is organized as a modular release, users may need to adjust local paths before executing the script in a different environment.

## Evaluation Script

The evaluation script loads the trained CC-STMT checkpoint and generates prediction results for the test period.

The main outputs include:

```text
test_predictions_expected.npy
test_predictions_mu.npy
test_predictions_pi.npy
test_predictions_theta.npy
test_predictions_v_hat.npy
test_ground_truth.npy
```

The expected prediction field is computed from the ZINB-related outputs. The `mu`, `pi`, and `theta` files preserve the probabilistic components of the prediction model. These files are important because the dispatching module uses the destination-side prediction fields to construct risk-aware matching costs.

The evaluation script may also produce prediction metrics such as:

```text
MAE
RMSE
MAPE on nonzero demand
SMAPE
R2
```

The exact output table depends on the city-specific script.

## Connection to DP-OTM

The prediction outputs generated by CC-STMT are used by the DP-OTM dispatching module. In the dispatching stage, each request has a destination zone. The prediction fields at the destination zone are used to estimate the future service condition after the vehicle completes the trip.

The main transferred quantities are:

```text
destination demand intensity
destination low-opportunity risk
destination demand dispersion
```

These quantities allow the dispatching module to distinguish between destinations with different post-drop-off service opportunities.

## Included and Excluded Content

The released CC-STMT module includes the proposed prediction model and the city-specific scripts needed to prepare, train, and evaluate it.

The released CC-STMT module does not include external forecasting baseline implementations. It also does not include large public raw datasets or unnecessary temporary files.

## Notes on Use

The module should be used as an implementation reference for the prediction component of the proposed framework. To reproduce the full prediction workflow, users should prepare the public raw data, spatial files, and contextual data described in the manuscript, then adapt the city-specific paths to their local environment.

The generated prediction outputs should be kept together with the processed city data because they are required by the DP-OTM dispatching module.
