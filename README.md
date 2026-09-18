# Freight Rate Prediction Challenge

Predicting `posted_rate` for freight loads from pickup/delivery city, distance, equipment type, weight, date, market index, and quote signal.

## The Data

- **Training set** (`data/train-test.csv`): 48,000 loads from January–October 2025, including the target `posted_rate`.
- **Validation set** (`data/validation.csv`): 12,000 loads from November 2025 onward, with no target  these are the loads that need predictions.
- **December inputs** (`data/december-chart-inputs.csv`): 31 rows, one per day of December 2025, with pickup/delivery/distance/equipment/weight held fixed (Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb) so only the date varies  used to chart the model's seasonal pricing pattern.

Each load record includes: `pickup`/`delivery` city, `pickup_lat`/`pickup_lon`/`delivery_lat`/`delivery_lon`, `distance` (miles), `equipment` (Dry Van, Reefer, or Flatbed), `weight` (lbs), `date`, `market_index`, `quote_signal`, and  in training only  `posted_rate`.

Distance is the strongest single predictor of rate (correlation ≈ 0.91). Equipment type also matters: Reefer loads average ~$2,554, Flatbed ~$2,445, Dry Van the lowest at ~$2,272. Rates drift over the year, rising into early summer, dipping mid summer, and recovering by fall  which is why validation is done chronologically rather than with a random split (see [notebooks/freight_exploratory-data-analysis.ipynb](notebooks/freight_exploratory-data-analysis.ipynb) for the full analysis).

## Data Quality Issues Found

- **Negative weights.** 292 rows in training (145 in validation) had a negative `weight`, which isn't physically possible  a load can't weigh less than zero. These were treated as sign errors rather than dropped.
- **Missing values.** `weight` was missing in 300 training rows (0.625%) and 165 validation rows (1.375%); `market_index` was missing in 374 training rows (0.78%) and 249 validation rows (2.08%). No other column had missing values.
- **Unseen cities.** 8 pickup cities and 8 delivery cities in validation never appear anywhere in training, so any encoding scheme has to tolerate categories it has never seen.
- **Distance vs. geography mismatches.** A handful of rows report a `distance` shorter than the true great circle distance between the two cities, which is impossible for a real route.
- No duplicate rows, no invalid `distance`/`market_index`/`quote_signal` values, and no missing values in location, equipment, date, or quote signal columns.

## How the Data Was Cleaned

Cleaning and feature preparation live in [script/freight_data_pipeline.py](script/freight_data_pipeline.py) as a fit/transform pipeline (`FreightFeaturePipeline`), so every rule is learned only from training data and then reapplied unchanged to validation and December data  nothing is ever recomputed on data the model shouldn't have seen yet:

1. **Negative weights** are converted to their absolute value rather than dropped, with a `weight_negative_flag` kept as a feature so the model still knows the row was originally suspect.
2. **Missing weight** is imputed using the equipment type median from training, falling back to the training wide median.
3. **Missing market index / quote signal** are imputed using a same date median first, then the training date median, then the training wide median as a last resort  always sourced from training data only.
4. **Coordinates** are cross checked against a canonical city coordinate table; mismatches beyond 50 miles are flagged (`coordinate_error_flag`) rather than silently trusted.
5. **Distance sanity** is checked against the true haversine distance between the two cities, flagging any route reported as shorter than physically possible (`distance_impossible_flag`).
6. Text fields (`pickup`, `delivery`, `equipment`) are whitespace normalized, and `date` is parsed to a proper datetime.
7. All rows are kept  nothing is dropped for being missing or invalid, since the issues found can be corrected or flagged without losing training examples.

## Models Used

`script/freight_model_training.py` compares three model families, each on both the raw `posted_rate` target and a `log1p` transformed target (since the rate distribution is right skewed), across several hyperparameter settings  25 candidates in total:

- **Ridge regression**  a linear baseline. Cheap to fit and easy to interpret, and useful as a sanity check on whether the relationship is even close to linear.
- **Extra Trees** (`ExtraTreesRegressor`)  an ensemble of randomized decision trees. Captures nonlinear effects and feature interactions (e.g. distance × equipment × route) without needing them to be specified by hand.
- **Histogram Gradient Boosting** (`HistGradientBoostingRegressor`)  a boosted tree ensemble. Generally the strongest of the three for tabular data with mixed numeric/categorical features, and fast enough to tune across many configurations.

Every candidate is scored using the same chronological cross validation described below, and the one with the lowest mean RMSE across folds is selected, refit on all training data, and used to generate the actual predictions. Log target models consistently outperformed raw target models (confirming the skew mattered), and both tree based families beat Ridge (confirming the relationship isn't purely linear).

### Top 3 Performing Models (by mean tuning RMSE)

| Rank | Candidate | Mean RMSE | Mean R² | Mean MAE |
|---|---|---|---|---|
| 1 | `hist_gradient_boosting__log__config_1` | 623.29 | 0.827 | 126.90 |
| 2 | `extra_trees__log__config_4` | 626.40 | 0.826 | 133.99 |
| 3 | `hist_gradient_boosting__log__config_4` | 626.85 | 0.825 | 137.25 |

The winner (#1) is Histogram Gradient Boosting on the log target with `max_iter=150, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.0`. On the untouched October holdout (never used for tuning) it scores RMSE 648.14, MAE 115.39, R² 0.820  the more trustworthy estimate of real-world performance. Full results for all 25 candidates are in `model_comparison.csv` after a training run.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

1. **Clean the data and engineer features** (reads from `data/`, writes to `freight_clean_data/`):

   ```bash
   python script/freight_data_pipeline.py --input-dir data --output-dir freight_clean_data
   ```

2. **Compare models and train the final one** (reads the prepared files, writes predictions and metrics to `freight_model_results/`):

   ```bash
   python script/freight_model_training.py \
       --data-dir freight_clean_data \
       --output-dir freight_model_results \
       --raw-data-dir data
   ```

   Add `--quick` for a faster run with a smaller hyperparameter grid (useful for testing the pipeline end-to-end before a full run).

3. **Validate the submission and render the December chart**:

   ```bash
   python score.py \
       --predictions freight_model_results/validation_predictions.csv \
       --december-predictions freight_model_results/december_chart_inputs.csv \
       --output-dir scorer_results
   ```

   This checks the prediction files match the required format and produces `scorer_results/candidate_december.png`.

## Project Structure

- `script/freight_data_pipeline.py`  leakage-safe data cleaning and feature engineering.
- `script/freight_model_training.py`  chronological model comparison and final training.
- `score.py`  validates a submission's format and renders the required December chart.
- `notebooks/freight_exploratory-data-analysis.ipynb`  the exploratory analysis behind the findings above.
