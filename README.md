# Risk-Aware Intelligent Dispatching for Ride-Hailing Systems

This repository provides the released implementation components for the paper:

**Risk-Aware Intelligent Dispatching for Ride-Hailing Systems Using Dual-Graph Attention Forecasting and Unbalanced Optimal Transport**

The repository contains two main code modules:

```text
code/
├── CC-STMT/
└── DP-OTM/
```

`CC-STMT` implements the prediction component of the proposed framework. `DP-OTM` implements the dispatching component that uses prediction outputs to support destination-aware online vehicle-request matching.

The code is organized as a modular implementation reference rather than a one-click executable package. The original experiments involve large public trip records, geospatial data, processed tensors, trained checkpoints, and server-specific directory structures. Therefore, users may need to adapt local file paths and prepare the required public datasets before using the scripts.

## Repository Structure

```text
.
├── code/
│   ├── CC-STMT/
│   │   ├── common/
│   │   ├── NYC/
│   │   ├── Chicago-A/
│   │   └── Chicago-B/
│   └── DP-OTM/
│       ├── common/
│       ├── NYC/
│       ├── Chicago-A/
│       ├── Chicago-B/
│       └── explain.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── .gitignore
└── README.md
```

The three city folders correspond to the experimental settings used in the study:

```text
NYC
Chicago-A
Chicago-B
```

The city-specific folders contain separate scripts because the three settings use different raw data sources, spatial zoning systems, temporal resolutions, and preprocessing assumptions.

## Method Overview

The proposed framework connects probabilistic spatio-temporal prediction with online ride-hailing dispatching.

The prediction module, CC-STMT, estimates future regional states from historical demand, traffic speed, weather or contextual information, en-route supply feedback, and dual graph structures. It produces demand intensity, low-opportunity risk, demand dispersion, and auxiliary speed prediction fields.

The dispatching module, DP-OTM, converts these prediction outputs into destination-aware vehicle-request matching costs. For each candidate assignment, the dispatching cost considers current service factors and predicted destination-side service opportunities. The final assignment is generated through an unbalanced optimal transport prior and a one-to-one assignment step.

This design allows the forecasting model to support online dispatching directly, rather than serving only as a standalone demand-count prediction model.

## CC-STMT Prediction Module

The prediction code is located in:

```text
code/CC-STMT/
```

The shared model definition is placed in:

```text
code/CC-STMT/common/
```

The city-specific prediction scripts are placed in:

```text
code/CC-STMT/NYC/
code/CC-STMT/Chicago-A/
code/CC-STMT/Chicago-B/
```

The CC-STMT module includes:

```text
data processing scripts
model training scripts
model evaluation scripts
shared CC-STMT model code
```

The processed prediction inputs generally include:

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

The prediction evaluation stage produces outputs such as:

```text
test_predictions_expected.npy
test_predictions_mu.npy
test_predictions_pi.npy
test_predictions_theta.npy
test_predictions_v_hat.npy
test_ground_truth.npy
```

These files provide the link between the prediction module and the dispatching module. In particular, the predicted demand intensity and low-opportunity risk fields are used by DP-OTM to evaluate request destinations.

External forecasting baseline implementations are not included in this repository.

## DP-OTM Dispatch Module

The dispatching code is located in:

```text
code/DP-OTM/
```

The shared dispatching logic is placed in:

```text
code/DP-OTM/common/
```

The city-specific dispatching scripts are placed in:

```text
code/DP-OTM/NYC/
code/DP-OTM/Chicago-A/
code/DP-OTM/Chicago-B/
```

The DP-OTM module includes:

```text
dispatching environment construction
order and vehicle state representation
mixed on-demand and advance request construction
advance-request visibility and activation logic
destination-aware matching cost construction
unbalanced optimal transport prior
one-to-one assignment step
dispatching result collection
component ablation settings
```

The dispatching cost considers:

```text
pickup distance
request revenue
waiting priority
destination opportunity
destination risk
```

The destination opportunity and destination risk terms are derived from the CC-STMT prediction outputs. This allows the dispatching model to consider not only the current pickup and request attributes, but also the future service condition of the request destination.

The released DP-OTM code may include internal component ablations for destination opportunity and destination risk. These ablations are part of the proposed framework analysis. External dispatching baseline implementations are not included in this repository.

## Data Sources

The experiments are based on public ride-hailing trip records and public spatial or contextual data. Large raw datasets are not stored directly in this repository.

The raw trip records used in the study are publicly available from:

```text
New York City Taxi and Limousine Commission trip record data portal
City of Chicago Transportation Network Providers trips data portal
```

The spatial and contextual data are obtained from public sources such as:

```text
City of Chicago Data Portal
OpenStreetMap
Open-Meteo
```

Users should download the corresponding raw data from the public sources described in the manuscript and adapt the local input paths before using the scripts.

## Processed Data and Results

Large processed tensors, trained checkpoints, and full experimental result files are not necessarily included in this repository. These files may be large and are dependent on the specific preprocessing configuration.

The expected processed files include prediction tensors, graph matrices, order tables, distance matrices, and prediction outputs. The dispatching module assumes that the relevant processed files and CC-STMT prediction outputs are available before dispatch simulation.

The repository may include small example files or result summaries when appropriate, but it is not intended to store large raw datasets or unnecessary intermediate files.

## Environment

Two environment files are provided:

```text
requirements.txt
environment.yml
```

`requirements.txt` can be used with pip-based environments. `environment.yml` can be used with conda-based environments.

The main dependencies include:

```text
PyTorch
NumPy
pandas
SciPy
scikit-learn
GeoPandas
Shapely
PyProj
Fiona
Pyogrio
Rtree
NetworkX
OSMnx
Requests
Joblib
Matplotlib
PyArrow
Fastparquet
```

CUDA is optional. If a compatible GPU is available and PyTorch is installed with CUDA support, the training scripts can use GPU acceleration.

## Reproducibility Notes

The released code is intended to support reproducibility of the proposed method and experimental design. Exact numerical reproduction requires consistency in:

```text
raw data version
spatial boundary files
road-network or distance data
weather data
preprocessing rules
temporal split
random seed
model hyperparameters
trained checkpoint
fleet size
request sampling rule
advance-request construction rule
dispatching time interval
matching cost weights
evaluation window
software environment
```

Because the full experimental pipeline depends on large public datasets and server-specific paths, users may need to modify local paths before applying the code in a new environment.

## Excluded Content

The repository intentionally excludes:

```text
external forecasting baseline code
external dispatching baseline code
large raw public datasets
large intermediate tensors
large baseline model weights
temporary cache files
notebook checkpoints
server-specific backup files
```

This keeps the repository focused on the proposed CC-STMT and DP-OTM framework.

## License

This repository is released under the Apache-2.0 License.

## Citation

If this repository or the implemented method is used in academic work, please cite the corresponding paper:

```text
Risk-Aware Intelligent Dispatching for Ride-Hailing Systems Using Dual-Graph Attention Forecasting and Unbalanced Optimal Transport
```
