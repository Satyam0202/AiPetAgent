# RetailPriceForecast

A small machine-learning portfolio project featuring three independent ML applications:

1. Employee Question-to-SQL Assistant
2. URL Risk Classification
3. Banana Ripeness Prediction

> **Note:** Despite the repository name, this project does not currently forecast retail prices. It contains three independent machine-learning demos with notebooks for data preparation, model training, and evaluation.

---

## Projects

| Demo                       | Description                                                                                                                                                          | Main Files                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Employee Query Assistant   | Classifies English questions about employee data, identifies missing information, and generates supported MySQL queries. Data-changing queries require confirmation. | `ask_query.py`, `build_query_datasets.py`, `ml.ipynb`, `mysql_setup.sql` |
| URL Risk Classifier        | Predicts whether a URL looks benign or potentially harmful using URL-text features. It does not open or visit the URL.                                               | `url_classifier.py`, `url.ipynb`, `data/url_dataset.csv`                 |
| Banana Ripeness Classifier | Classifies banana images into four categories: unripe, ripe, overripe, and rotten. It also provides approximate storage-day guidance.                                | `banana_classifier.py`, `banana.ipynb`, `data/banana_ripeness/`          |

---

## Repository Structure

```text
RetailPriceForecast/
│
├── ask_query.py
├── build_query_datasets.py
├── banana_classifier.py
├── banana.ipynb
├── banana_model.pt
├── url_classifier.py
├── url.ipynb
├── url_model.joblib
├── ml.ipynb
├── mysql_setup.sql
│
└── data/
    ├── intent_dataset.csv
    ├── ambiguity_dataset.csv
    ├── url_dataset.csv
    │
    └── banana_ripeness/
        ├── train/
        ├── valid/
        └── test/
```

The employee intent and ambiguity datasets are generated using synthetic templates and augmented examples. They do not represent real user requests.

---

## Requirements

* Python 3.10 or later
* Jupyter Notebook
* MySQL Server (only required for the Employee Query Assistant)

### Install Dependencies

Run the following command:

```bash
python -m pip install numpy pandas scikit-learn joblib Pillow torch matplotlib mysql-connector-python jupyter
```

**Note:** PyTorch installation may vary depending on your operating system and hardware. If the installation fails, use the official installation guide:

[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>

cd RetailPriceForecast
```

### 2. Employee Question-to-SQL Assistant

This application converts natural-language employee questions into supported MySQL queries.

**Setup:**

1. Start your MySQL Server.
2. Open MySQL Workbench.
3. Execute the `mysql_setup.sql` file.
4. This creates a database named `company_db` and an `employees` table containing 500 sample records.

**Run the application:**

```bash
python ask_query.py
```

**Example question:**

```text
Show employees in the IT department
```

The application classifies the question, checks for missing information, and generates a supported SQL query.

To stop the application, enter:

```text
quit
```

You can also use `exit` or `q`.

The assistant trains its text classifiers using the training rows from:

* `data/intent_dataset.csv`
* `data/ambiguity_dataset.csv`

**MySQL Configuration**

The default database connection settings are:

| Variable         | Purpose           | Default                        |
| ---------------- | ----------------- | ------------------------------ |
| `MYSQL_HOST`     | MySQL server host | `127.0.0.1`                    |
| `MYSQL_PORT`     | MySQL server port | `3306`                         |
| `MYSQL_USER`     | MySQL username    | Prompted; falls back to `root` |
| `MYSQL_PASSWORD` | MySQL password    | Prompted securely              |

---

### 3. URL Risk Classifier

The URL classifier analyzes URL-text features and predicts whether a URL appears benign or potentially harmful.

**Run a prediction:**

```bash
python -c "from url_classifier import predict_url; print(predict_url('https://example.com/account'))"
```

**Model file:**

```text
url_model.joblib
```

**Retrain the model:**

1. Open `url.ipynb` in Jupyter Notebook.
2. Run the notebook cells from top to bottom.
3. The notebook prepares the dataset, trains the classifier, and evaluates its performance.

If the URL dataset is not available locally, the notebook downloads it.

---

### 4. Banana Ripeness Classifier

The banana classifier uses image-based machine learning to identify the ripeness stage of a banana.

It supports four classes:

* Unripe
* Ripe
* Overripe
* Rotten

**Run a prediction:**

```bash
python -c "from banana_classifier import predict_banana; print(predict_banana('path/to/banana.jpg'))"
```

Replace `path/to/banana.jpg` with the actual image path.

**Model file:**

```text
banana_model.pt
```

**Retrain the model:**

1. Open `banana.ipynb` in Jupyter Notebook.
2. Run all cells from top to bottom.
3. The notebook prepares the image dataset, trains the model, and evaluates its performance.
4. The trained model is saved as `banana_model.pt`.

The classifier also provides approximate storage-day guidance based on the predicted ripeness class.

---

## Employee Dataset Generation

The employee question datasets can be regenerated using:

```bash
python build_query_datasets.py
```

This generates two CSV files:

```text
data/
├── intent_dataset.csv
└── ambiguity_dataset.csv
```

These datasets contain synthetic employee-related questions used to train the intent and ambiguity classifiers.

---

## Data Sources and Attribution

### Banana Image Dataset

The banana ripeness images are sourced from:

[Banana Ripeness Classification – Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification)

The dataset is provided under the **CC BY 4.0** license.

The notebook downloads a pinned dataset archive from:

[Banana Ripeness Dataset – Hugging Face](https://huggingface.co/datasets/Sapek007/banana-ripeness)

### URL Dataset

The URL classifier uses the publicly available `malicious_phish.csv` dataset from:

[URL Dataset – GitHub](https://github.com/SahinMuhammetAbdullah/urldataset)

Check the original dataset's terms and licensing conditions before redistribution.

### Employee Query Dataset

The employee intent and ambiguity datasets are generated locally using synthetic templates through `build_query_datasets.py`.

---

## Important Limitations

### URL Risk Classifier

* Predictions are estimates, not a guarantee of website safety.
* The classifier analyzes URL text only.
* It does not open websites, follow redirects, or verify real-time URL reputation.
* It should not be used as the only security mechanism.

### Banana Ripeness Classifier

* Predictions represent visual ripeness categories, not exact expiry dates.
* Storage-day estimates are approximate.
* Actual banana freshness depends on temperature, humidity, and storage conditions.
* Model confidence is not calibrated.

### Employee Query Assistant

* SQL generation is limited to the employee table and query patterns implemented in `ask_query.py`.
* The application is designed for demonstration purposes.
* Generated SQL should be reviewed before using the system with real employee data.
* Data-changing queries require user confirmation.

---

## Running Jupyter Notebooks

To explore the model training and evaluation workflows, start Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Available notebooks:

| Notebook       | Purpose                                                   |
| -------------- | --------------------------------------------------------- |
| `ml.ipynb`     | Employee intent classification and ambiguity detection    |
| `url.ipynb`    | URL feature extraction, model training, and evaluation    |
| `banana.ipynb` | Image dataset preparation, model training, and evaluation |

**Important:** Always run the notebooks from the repository root directory so that relative `data/` paths work correctly.

---

## Project Summary

| Project                    | Machine Learning Task | Output                   |
| -------------------------- | --------------------- | ------------------------ |
| Employee Query Assistant   | Text Classification   | Supported MySQL Queries  |
| URL Risk Classifier        | Binary Classification | URL Risk Prediction      |
| Banana Ripeness Classifier | Image Classification  | Banana Ripeness Category |

This repository demonstrates practical applications of machine learning in natural-language processing, URL classification, and computer vision.
