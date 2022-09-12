
# Flight price prediction (MLOps Zoomcamp 2022 project)

## The scope of the proje   ct

Airline ticket purchasing is challenging topic because buyers have insufficient information for reasoning about pricing. Prices are changing between sellers, airlines, seasons, time remaining and so on. Thus, it is interesting to simulate how various airlines prices are varying and predict possible value depending on selected criteria. 

The project is a very basic implementation of solution to the problem. Just simple statistical reasoning is involved. In future solutions more precise datasets should be included and more elaborated models should be prepared. E.g. price changes are depending not only on airline, source and destination town, but also on the time of departure. 

The problem and solution are to some extend inspired by this [Kaggle competition](https://www.kaggle.com/code/anshigupta01/flight-price-prediction).

The original dataset was provided in XLSX format; for demonstration purposes here we use the same data transferred to PARQUET format.

## Getting started

1. Clone the repository to the local or remote machine

2. Be sure there is a `res` folder with `flight_data_train.parquet`. If not, go to the mentioned Kaggle competition, find a Excel file and just simply transfer it to parquet file (e.g. with Pandas).

3. It is highly recommended to use virtual environment before any further steps (venv, conda, pipenv, etc.). Examples of venv creation:

with `venv`:

```
shell
python3 -m venv venv
source venv/bin/activate
```


with `pipenv`:

```
shell
pip install pipenv
pipenv install
pipenv shell
```

4. With acitve virtual env run `core.py`. There are values for grid search of best parameters provided in the file, but feel free to make changes to the dictionary.

```shell
python core.py
```

The function will search for the best model and will save it as a `model.bin` file in a ``prediction_service` folder. It will be done automatically but you can check result of modelling process by hand starting mlflow server, e.g.:

```
shell
mlflow-env % mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each experiment and registered model can be easily accessed from UI:

![mlflow experiments](/res/images/mlflow_experiments.png)

![mlflow models](/res/images/mlflow_models.png)

    - function `core.py` is preparing and saving model to the file together with DictVecorizer and MultiColumnLabelEncoder. The later class function takes care of all categorical variables in the process of model creation: each column in a dataframe is categorized into numerical value and the information about classes is stored for later. Classes information will be necessary in order to translate labels that an user can select when interacting with application. In other words when application will be deployed user will be able to provide classes like 'Air Asia' or 'Bangalore', and the encoder will consistently translate them to numbers.

6. Now we can move to creating interactive contenerised application. This is a Flask application combined with Evidently monitoring services application. The application is prepared so with running following command we can request for predictions (predicted prices based on provided information about the carrier (airline), source and destination towns and number of stops on the way. Currently user have to provide just numbers but in the next release of the application interactive app with drop down menus will take care of translation labels into number which are understandable by model.


