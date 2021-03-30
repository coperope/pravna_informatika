# PI

### Authors
- Mihajlo Perendija @mihajlo-perendija
- Petar Bašić @coperope

## Getting started
### Prerequisites
Ensure you have installed/downloaded the following before proceeding further:

- Python 3.7 
- Installed other requirements
### Original data
Original dataset for this project can be found at http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/index.html
This data consists of large amount of `.json` files. For easier use that data has been combined into `.csv` files available at <link>

## Instructions

### Training & testing models
#### Preparing data
Because of the GitHub restrictions for file sizes required dataset, vectors etc. have to be downloaded manually.

Directory labs contains prepared notebooks for training models: TF-IDF, Word2Vec, other_tested_methods.ipynb (which contains code for training ML-KNN, Gradient boosting & SVC models), legal_bert_multi_label_classification and lstm. 
#### Train & test models from scratch
After cloning the repository do the following:
For all models:
- From <link> download prepared datasets (.csv files) and place them in the `/data/dataset/` directory.

For Word2Vec method additionaly:
- From <link> download pretrained law2vec models and place them in the `/labs/Law2Vec/` directory.

Open desired notebook e.g. via jupyter lab & execute all cells.

#### Train & test models with prepared vectors
After cloning the repository do the following:
From <link> download vectorized documents & concepts (all .npz files) and place them in the `/labs/vectors/` directory.
Open desired notebook e.g. via jupyter lab & execute all cells except cells marked for vectorizing (with comment `#FOR_VECTORIZING`)

### Running pre-trained model on fresh data
To quickly test the work of the model which had the best results (method that uses TF-IDF vectorizing & classic neural network)
we have prepaired a python script and placed it in the `/app/tf-idf-nn.py`.
To run the script there has to be a pre-trained model placed in the `/app` directory. Model can be downloaded from <link>.

Some test data has been prepared and placed in the `/app/test_files` directory. Feel free to run the script on some other data which can be found in the original dataset.
Run the following to execute the script:
```
python tf-idf-nn.py -i <name_of_the_file_from_test_files_dir.json>
# Example:
python tf-idf-nn.py -i 32015R0576.json
```
