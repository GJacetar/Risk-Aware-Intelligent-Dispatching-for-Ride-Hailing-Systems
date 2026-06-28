# DP-OTM Dispatch Module Description

The `DP-OTM` directory contains the dispatching component of the proposed framework. DP-OTM stands for Demand-Prediction-guided Optimal Transport Matching. It is designed to generate online vehicle-request assignments under mixed on-demand and advance request settings.

The dispatch module is organized as follows:

```text
DP-OTM/
├── common/
├── NYC/
├── Chicago-A/
├── Chicago-B/
└── explain.md
```

The `common` folder contains the shared dispatching logic. The city folders contain city-specific dispatching entry files and configuration settings.

## Module Purpose

The purpose of the DP-OTM module is to convert prediction outputs into executable dispatching decisions. The module evaluates each candidate vehicle-request pair using both current service factors and future destination-side predictive factors.

The dispatching cost considers:

```text
pickup distance
request revenue
waiting priority
destination opportunity
destination risk
```

The destination opportunity and destination risk terms are derived from the CC-STMT prediction outputs. This design allows each dispatching decision to account for both the current trip and the vehicle state after drop-off.

## Shared Dispatch Code

The shared dispatching code is placed in:

```text
DP-OTM/common/
```

This folder contains the common environment and algorithmic logic used by all city settings.

The shared code includes:

```text
order state representation
vehicle state representation
mixed-request environment construction
on-demand request activation
advance request visibility and activation
vehicle availability update
destination opportunity and risk extraction
matching cost construction
unbalanced optimal transport prior
one-to-one linear assignment
dispatching result collection
component ablation logic
```

The shared code is independent of a specific city as long as the processed data and prediction outputs follow the expected format.

## City-Specific Dispatch Files

The dispatch module includes three city-specific folders:

```text
DP-OTM/NYC/
DP-OTM/Chicago-A/
DP-OTM/Chicago-B/
```

Each city folder provides the city-specific dispatching interface. These files define or reference the corresponding city paths, temporal settings, fleet settings, processed order files, and output directories.

The city-specific files are separated because the three experimental settings differ in:

```text
spatial zoning system
prediction time interval
dispatching time interval
processed order table format
fleet size setting
request sampling setting
city-specific data path
```

## Required Inputs

The DP-OTM module should be used after the CC-STMT prediction module has produced its outputs.

The required city-level inputs include:

```text
processed order table
distance matrix
zone mapping file
CC-STMT prediction outputs
```

Typical files include:

```text
dist_matrix.npy
zone_mapping.csv
test_predictions_expected.npy
test_predictions_mu.npy
test_predictions_pi.npy
test_predictions_theta.npy
test_predictions_v_hat.npy
test_ground_truth.npy
```

The processed order table should provide enough information to construct request states. Required fields include pickup zone, drop-off zone, request or pickup time slot, trip duration, trip distance, and revenue or fare.

## Mixed-Request Construction

The dispatching environment supports mixed on-demand and advance requests.

On-demand requests are assumed to enter the dispatching pool when they are created. Their waiting priority increases as they remain unserved.

Advance requests are visible before the intended pickup time but are not necessarily assigned immediately. They become dispatchable only within a short executable service window around the requested pickup time. This design separates early visibility from executable vehicle commitment.

Each request is represented by a unified order state containing:

```text
request identifier
pickup zone
drop-off zone
target pickup slot
trip duration
revenue
trip distance
creation slot
request type
service window information
```

This unified representation allows on-demand and advance requests to be handled within the same matching process.

## Vehicle State Construction

Each vehicle is represented by a state that includes:

```text
vehicle identifier
current zone
next available time
accumulated revenue
accumulated empty travel distance
served request count
```

After a vehicle is assigned to a request, its state is updated according to the empty travel distance, pickup timing, trip duration, and destination zone. The updated destination zone becomes the vehicle location after completing the trip.

## DP-OTM Matching Logic

At each dispatching epoch, the environment constructs a set of available vehicles and a set of dispatchable requests. For each feasible vehicle-request pair, the algorithm builds a composite matching cost.

The cost includes current operational terms and destination-side predictive terms. Pickup distance penalizes long empty movement. Revenue rewards economically valuable requests. Waiting priority increases the priority of requests that are close to or beyond their service time. Destination opportunity rewards destinations with stronger predicted future service potential. Destination risk penalizes destinations with higher predicted low-opportunity risk.

After constructing the cost matrix, the algorithm first computes an unbalanced optimal transport prior. This prior provides a soft matching structure under imbalanced vehicle and request supply. The final executable assignment is then obtained through a one-to-one linear assignment step.

The final result satisfies:

```text
each vehicle is assigned to at most one request
each request is assigned to at most one vehicle
only feasible vehicle-request pairs can be assigned
```

## Component Ablation Settings

The released dispatch code may include internal component ablation settings. These settings are used to examine the effect of destination opportunity and destination risk inside the proposed dispatching framework.

The component settings include:

```text
without destination opportunity and destination risk
with destination opportunity only
with destination risk only
with both destination opportunity and destination risk
full DP-OTM
```

These settings are internal ablations of the proposed method. They are not external dispatching baseline algorithms.

External dispatching comparison algorithms are not included in this released code.

## Main Outputs

The dispatching module records the operational outcomes of the online simulation. The main indicators include:

```text
served requests
rejected requests
rejection rate
total revenue
empty travel distance
waiting time
destination opportunity
destination risk
post-drop-off low-opportunity indicators
runtime
```

The result files can be used to support the dispatching performance analysis and component ablation analysis in the manuscript.

## Relationship to CC-STMT

DP-OTM depends on the prediction outputs generated by CC-STMT. The prediction module estimates future regional demand and risk states. The dispatching module maps these regional states to request-level destination scores according to each request's drop-off zone.

This connection is the central prediction-dispatching link of the framework. The prediction model does not only produce standalone accuracy metrics; its outputs directly influence the online matching cost.

## Notes on Use

The dispatch module is released as a structured implementation reference. Since the code has been separated into common and city-specific files, users should check the local data paths and file names before using it in a new environment.

The module is not intended to be a one-click runnable package without prepared inputs. It assumes that the processed city data and CC-STMT prediction outputs already exist.

For full experimental reproduction, users should keep the following settings consistent with the manuscript:

```text
processed trip records
prediction outputs
dispatching time interval
fleet size
request sampling rule
advance-request visibility rule
advance-request activation window
pickup distance constraint
cost weights
random seed
evaluation window
```

The released code focuses on the proposed DP-OTM framework and its component ablations.
