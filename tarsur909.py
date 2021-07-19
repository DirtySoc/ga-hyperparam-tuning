# %% Imports
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Read in dataset.
data = pd.read_csv("impstroke.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)
data.head()

# %% Split data
X = data.drop("stroke", axis=1)
y = data["stroke"]
X.head()
# y.head()

# %% Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=97)
X_train.head()

# %%
model = xgb.XGBClassifier(verbosity=0)
model.fit(X_train, y_train)
print("f1_score:", f1_score(y_test, model.predict(X_test)))
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# %% GA Code

random.seed(723)
np.random.seed(723)


def initilialize_poplulation(numberOfParents):
    learningRate = np.empty([numberOfParents, 1])
    nEstimators = np.empty([numberOfParents, 1], dtype=np.uint8)
    maxDepth = np.empty([numberOfParents, 1], dtype=np.uint8)
    minChildWeight = np.empty([numberOfParents, 1])
    gammaValue = np.empty([numberOfParents, 1])
    subSample = np.empty([numberOfParents, 1])
    colSampleByTree = np.empty([numberOfParents, 1])

    for i in range(numberOfParents):
        # print(i)
        learningRate[i] = round(random.uniform(0.01, 1), 2)
        nEstimators[i] = random.randrange(10, 1500, step=25)
        maxDepth[i] = int(random.randrange(1, 10, step=1))
        minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
        gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
        subSample[i] = round(random.uniform(0.01, 1.0), 2)
        colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)

    population = np.concatenate((learningRate, nEstimators, maxDepth,
                                minChildWeight, gammaValue, subSample, colSampleByTree), axis=1)
    return population


def fitness_score(y_true, y_pred):
    fitness = round((f1_score(y_true, y_pred)), 4)
    # fitness = round((accuracy_score(y_true, y_pred)), 4)
    return fitness


def train_population(population, dMatrixTrain, dMatrixtest, y_test):
    aScore = []
    for i in range(population.shape[0]):
        param = {'objective': 'binary:logistic',
                 'learning_rate': population[i][0],
                 'n_estimators': population[i][1],
                 'max_depth': int(population[i][2]),
                 'min_child_weight': population[i][3],
                 'gamma': population[i][4],
                 'subsample': population[i][5],
                 'colsample_bytree': population[i][6],
                 'seed': 24}
        num_round = 100
        xgbT = xgb.train(param, dMatrixTrain, num_round)
        preds = xgbT.predict(dMatrixtest)
        preds = preds > 0.5
        aScore.append(fitness_score(y_test, preds))
    return aScore


def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1]))
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1
    return selectedParents


def crossover_uniform(parents, childrenSize):

    crossoverPointIndex = np.arange(
        0, np.uint8(childrenSize[1]), 1, dtype=np.uint8)
    crossoverPointIndex1 = np.random.randint(
        0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1]/2))
    crossoverPointIndex2 = np.array(
        list(set(crossoverPointIndex) - set(crossoverPointIndex1)))

    children = np.empty(childrenSize)

    for i in range(childrenSize[0]):
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        children[i, crossoverPointIndex1] = parents[parent1_index,
                                                    crossoverPointIndex1]
        children[i, crossoverPointIndex2] = parents[parent2_index,
                                                    crossoverPointIndex2]
    return children


def mutation(crossover, numberOfParameters, selectedParentsStats):
    minMaxValue = np.zeros((numberOfParameters, 2))

    minMaxValue[0:] = [0.01, 1.0]
    minMaxValue[1, :] = [10, 2000]
    minMaxValue[2, :] = [1, 15]
    minMaxValue[3, :] = [0, 10.0]
    minMaxValue[4, :] = [0.01, 10.0]
    minMaxValue[5, :] = [0.01, 1.0]
    minMaxValue[6, :] = [0.01, 1.0]

    mutationValue = 0
    parameterSelect = np.random.randint(0, 7)
    mutationValue = round(np.random.normal(selectedParentsStats['mean'][parameterSelect], selectedParentsStats['sd'][parameterSelect]))
    # print(parameterSelect)
    # if parameterSelect == 0:
    #     mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    # if parameterSelect == 1:
    #     mutationValue = np.random.randint(-200, 200, 1)
    # if parameterSelect == 2:
    #     mutationValue = np.random.randint(-5, 5, 1)
    # if parameterSelect == 3:
    #     mutationValue = round(np.random.uniform(-5, 5), 2)
    # if parameterSelect == 4:
    #     mutationValue = round(np.random.uniform(-2, 2), 2)
    # if parameterSelect == 5:
    #     mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    # if parameterSelect == 6:
    #     mutationValue = round(np.random.uniform(-0.5, 0.5), 2)

    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx,
                                                    parameterSelect] + mutationValue
        if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
        if(crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]
    return crossover


# %%
xgDMatrix = xgb.DMatrix(X_train, y_train)
xgbDMatrixTest = xgb.DMatrix(X_test, y_test)

numberOfParents = 64
numberOfParentsMating = 32
numberOfParameters = 7
numberOfGenerations = 10

populationSize = (numberOfParents, numberOfParameters)
population = initilialize_poplulation(numberOfParents)
# print(population)
fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents])
populationHistory = np.empty(
    [(numberOfGenerations+1)*numberOfParents, numberOfParameters])
populationHistory[0:numberOfParents, :] = population

# feature_statistics = {
#   'mean': [],
#   'sd': []
# }
for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))

    fitnessValue = train_population(
        population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
    fitnessHistory[generation, :] = fitnessValue

    print('Best F1 score in the this iteration = {}'.format(
        np.max(fitnessHistory[generation, :])))
    # print('Best Accuracy score in the this iteration = {}'.format(
    #     np.max(fitnessHistory[generation, :])))

    parents = new_parents_selection(
        population=population, fitness=fitnessValue, numParents=numberOfParentsMating)
    # print(parents)
    # feature_statistics['mean'].append(np.mean(parents, axis=0))
    # feature_statistics['sd'].append(np.std(parents, axis=0))

    feature_statistics = {
      'mean': np.mean(parents, axis=0),
      'sd': np.std(parents, axis=0)
    }

    children = crossover_uniform(parents=parents, childrenSize=(
        populationSize[0] - parents.shape[0], numberOfParameters))
    children_mutated = mutation(children, numberOfParameters, feature_statistics)

    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = children_mutated
    populationHistory[(generation+1)*numberOfParents: (generation+1)
                      * numberOfParents + numberOfParents, :] = population

fitness = train_population(
    population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
fitnessHistory[generation+1, :] = fitness
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

print("Best fitness is =", fitness[bestFitnessIndex])
print("Best parameters are:")
print('learning_rate', population[bestFitnessIndex][0])
print('n_estimators', population[bestFitnessIndex][1])
print('max_depth', int(population[bestFitnessIndex][2]))
print('min_child_weight', population[bestFitnessIndex][3])
print('gamma', population[bestFitnessIndex][4])
print('subsample', population[bestFitnessIndex][5])
print('colsample_bytree', population[bestFitnessIndex][6])

# %%
