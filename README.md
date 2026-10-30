# RetailPriceForecast

> **A small machine-learning portfolio project:** employee questions to SQL, URL risk classification, and banana ripeness prediction.

Despite the repository name, the current code does **not** forecast retail prices. It contains three independent ML demos and notebooks for exploring and training them.

## Projects

| Demo | What it does | Main files |
|---|---|---|
| Employee query assistant | Classifies an English employee-data question, asks for missing details, and builds a supported MySQL query. Data-changing queries require confirmation. | `ask_query.py`, `build_query_datasets.py`, `ml.ipynb`, `mysql_setup.sql` |
| URL risk classifier | Estimates whether a URL looks benign or harmful from URL-text features. It does not open or visit the URL. | `url_classifier.py`, `url.ipynb`, `data/url_dataset.csv` |
| Banana ripeness classifier | Predicts one of four image classes—unripe, ripe, overripe, or rotten—and gives a rough storage-day guide. | `banana_classifier.py`, `banana.ipynb`, `data/banana_ripeness/` |

## Repository layout

```text
.
├── ask_query.py                 # Employee question-to-SQL assistant
├── build_query_datasets.py      # Creates the intent and ambiguity CSV datasets
├── banana_classifier.py         # Loads the banana image model and predicts a class
├── banana.ipynb                # Banana dataset preparation, training, and evaluation
├── banana_model.pt              # Saved banana model
├── url_classifier.py           # URL feature extraction and prediction
├── url.ipynb                    # URL dataset preparation, training, and evaluation
├── url_model.joblib             # Saved URL model
├── ml.ipynb                     # Employee intent and ambiguity model exploration
├── mysql_setup.sql              # Creates the demo employee database and table
└── data/
    ├── intent_dataset.csv
    ├── ambiguity_dataset.csv
    ├── url_dataset.csv
    └── banana_ripeness/         # Image dataset (train/valid/test)
```

The employee intent and ambiguity datasets are generated from templates and augmented examples; they are not a collection of real user requests.

## Requirements

- Python 3.10 or later
- Jupyter Notebook (to run the notebooks)
- MySQL Server (only for the employee query assistant)

Install the Python packages:

```bash
python -m pip install numpy pandas scikit-learn joblib Pillow torch matplotlib mysql-connector-python jupyter
```

PyTorch installation can differ by operating system and hardware. If the command above does not install a compatible build, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Getting started

Clone the repository and enter its folder:

```bash
git clone <your-repository-url>
cd RetailPriceForecast
```

### 1. Employee question-to-SQL assistant

1. Start MySQL Server.
2. Run `mysql_setup.sql` in MySQL Workbench (or another MySQL client). It creates `company_db` and a sample `employees` table with 500 records.
3. From the repository root, start the assistant:

   ```bash
   python ask_query.py
   ```

4. Ask a question about employees, for example:

   ```text
   Show employees in the IT department
   ```

   Enter `quit`, `exit`, or `q` to stop. The assistant trains its text classifiers from the **training rows** in `data/intent_dataset.csv` and `data/ambiguity_dataset.csv` when it starts.

The connection defaults to `127.0.0.1:3306` and database `company_db`. Set these environment variables to use a different local MySQL account:

| Variable | Purpose | Default |
|---|---|---|
| `MYSQL_HOST` | MySQL server host | `127.0.0.1` |
| `MYSQL_PORT` | MySQL server port | `3306` |
| `MYSQL_USER` | MySQL username | Prompted; falls back to `root` |
| `MYSQL_PASSWORD` | MySQL password | Prompted securely |

### 2. URL risk classifier

The saved model is loaded by `url_classifier.py`. From the repository root, try a prediction with:

```bash
python -c "from url_classifier import predict_url; print(predict_url('https://example.com/account'))"
```

To retrain or inspect the workflow, open `url.ipynb` in Jupyter and run its cells from top to bottom. If the URL dataset is not already present, the notebook downloads it.

### 3. Banana ripeness classifier

The saved model is loaded by `banana_classifier.py`. Pass it the path to a banana image:

```bash
python -c "from banana_classifier import predict_banana; print(predict_banana('path/to/banana.jpg'))"
```

To retrain or inspect the workflow, open `banana.ipynb` in Jupyter and run its cells from top to bottom. The notebook prepares the image dataset and saves `banana_model.pt`.

### Employee dataset generation

To regenerate the synthetic employee question datasets:

```bash
python build_query_datasets.py
```

This writes `data/intent_dataset.csv` and `data/ambiguity_dataset.csv`.

## Data sources and attribution

- **Banana images:** [Banana Ripeness Classification on Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification), provided under **CC BY 4.0**. The notebook downloads a pinned dataset archive from [Hugging Face](https://huggingface.co/datasets/Sapek007/banana-ripeness).
- **URL data:** the notebook downloads the public `malicious_phish.csv` dataset from [SahinMuhammetAbdullah/urldataset](https://github.com/SahinMuhammetAbdullah/urldataset). Check the source dataset's terms before redistribution.
- **Employee query data:** generated locally by `build_query_datasets.py` from synthetic templates.

## Important limitations

- **URL predictions are estimates, not a safety guarantee.** The classifier analyzes URL text and does not fetch the page, follow redirects, or verify its current reputation. Do not rely on it as the sole security check.
- **Banana predictions are visual classes, not measured expiry dates.** The displayed day ranges are rough room-temperature guidance; storage conditions matter, and model confidence is not calibrated.
- **Employee SQL support is intentionally limited** to the employee table and query patterns implemented in `ask_query.py`. Review the generated result before using this demo with non-sample data.

## Notebooks

Run notebooks from the repository root so their relative `data/` paths resolve correctly:

```bash
jupyter notebook
```

The notebooks contain the data preparation, training, and evaluation workflows for the corresponding demos.
