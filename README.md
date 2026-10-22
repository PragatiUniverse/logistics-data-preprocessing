
# Logistics Data Collection, Cleaning and Preprocessing

**Author:** Pragati  
**Course:** B.Tech

## Objective
To simulate a logistics data-preprocessing pipeline using Python and prepare data for further analysis.

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- VS Code

## Dataset
The raw CSV is synthetic practice data with 503 rows and 15 original columns. It includes examples of missing values, duplicate rows, inconsistent labels, invalid numeric values and malformed dates.

## Workflow
1. Load and inspect the raw dataset.
2. Standardize text and parse dates.
3. Remove exact duplicate rows.
4. Validate numeric ranges.
5. Flag potential outliers.
6. Handle missing values using field-specific rules.
7. Create delivery-performance features.
8. Normalize selected numeric columns.
9. Export the cleaned dataset and quality summary.

## How to Run
Install the requirements:

```bash
python -m pip install -r requirements.txt
```

Run the preprocessing script:

```bash
python preprocess.py
```

## Output Files
- `data/logistics_orders_cleaned.csv`
- `data/data_quality_summary.csv`

## Limitation
The dataset is synthetic and intended for educational purposes. It does not represent actual logistics-company performance.
