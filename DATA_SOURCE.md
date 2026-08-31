# Data Source

## Dataset
Telco Customer Churn benchmark from OpenML, data id `42178`.

## Expected structure
- 7,043 customer rows
- target: `Churn`
- identifier removed from modeling: `customerID`
- remaining model inputs: 19 columns

## Data-quality handling
`TotalCharges` is coerced to numeric. Non-numeric values are treated as missing and filled with `0.0`, matching the zero-tenure/no-accumulated-charge interpretation used only as a reproducible benchmark convention. The pipeline records how many values were coerced.

## Integrity controls
The loader fails on unexpected row count, missing target/identifier columns, duplicate customer IDs, or unknown target labels. It also records a SHA-256 fingerprint of the cached frame.

## Evaluation boundary
The benchmark is randomly stratified into 60% training, 20% validation, and 20% locked test. This is not temporal or out-of-time validation and should not be presented as evidence of live telecom deployment performance.
