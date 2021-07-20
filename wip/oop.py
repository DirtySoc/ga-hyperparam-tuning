# %% Imports
from oop import Population
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import xgboost as xgb

random.seed(723)
np.random.seed(723)

# %% Constants

POP_SIZE = 64
NUM_GENERATION = 10
NUM_MATING_PARENTS = 32

# %% Define Model Class


class Model:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'n_estimators': random.randrange(10, 1500, step=25),
                'learning_rate': round(random.uniform(0.01, 1), 2),
                'maxDepth': int(random.randrange(1, 20, step=1)),
                'minChildweight': round(random.uniform(0.01, 10.0), 2),
                'gammaValue': round(random.uniform(0.01, 10.0), 2),
                'subSample': round(random.uniform(0.01, 1.0), 2),
                'colSampleByTree': round(random.uniform(0.01, 1.0), 2),
                'seed': 24
            }
        self.params = params
        self.num_round = 100

    def train(self, xgb_dmatrix_train):
        self.trained_xgb = xgb.train(
            self.params, xgb_dmatrix_train, self.num_round)

    def predict(self, xgb_dmatrix_test):
        preds = self.trained_xgb.predict(xgb_dmatrix_test)
        self.preds = preds > 0.5

    def calc_fitness(self, y_test):
        self.fitness = round(f1_score(y_test, self.preds), 4)

# %% Define Population Class


class Population:
    def __init__(self, size, X_train, y_train, X_test, y_test):
        self.generation = 0
        self.population = []
        self.history = []
        self.fitness_history = []
        self.y_test = y_test
        self.xgb_dmatrix_train = xgb.DMatrix(X_train, y_train)
        self.xgb_dmatrix_test = xgb.DMatrix(X_test, y_test)
        for i in range(size):
            self.population.append(Model())

    def reproduce(self, n_parents):
        self.fitness_history.append(self.fitness)
        self.history.append(self.population)
        
        # Select Parents based on max fitness
        selectedParents = []
        for i in range(n_parents):
            most_fit_idx = np.where(self.fitness == np.max(self.fitness))[0][0]
            selectedParents.append(self.population[most_fit_idx])
            self.fitness[most_fit_idx] = -1
        
        # Crossover
        children = []


        print('DEBUG')

        # Mutate

        self.fitness = []
        self.generation += 1
        return NotImplemented

    def train(self):
        for model in self.population:
            model.train(self.xgb_dmatrix_train)

    def calc_fitness(self):
        pop_fitness = []
        for model in self.population:
            model.predict(self.xgb_dmatrix_test)
            model.calc_fitness(y_test)
            pop_fitness.append(model.fitness)
        self.fitness = pop_fitness


# %% Read in Dataset
data = pd.read_csv("impstroke.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)
X = data.drop('stroke', axis=1)
y = data['stroke']

# %% Split dataset into test/train splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=97)

# %% View dataset
X_train.head()

# %% Test the default XGBoost Classifier
baseline = xgb.XGBClassifier(use_label_encoder=False, verbosity=0)
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
print("F1 Score: ", f1_score(y_test, pred))
print("Accuracy Score: ", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# %% Define Genetic Algorithm Functions


def init_population(n_parents):
    population = []
    for i in range(n_parents):
        population.append(Model())
    return population


# %% Main Function
if __name__ == '__main__':
    population = Population(POP_SIZE, X_train, y_train, X_test, y_test)

    while(population.generation < NUM_GENERATION):
        print('This is generation ', population.generation)
        population.train()
        population.calc_fitness()
        print("Best fitness in this generation: ", np.max(population.fitness))
        population.reproduce(NUM_MATING_PARENTS)
        
        print("DEBUG")

    print('END')
