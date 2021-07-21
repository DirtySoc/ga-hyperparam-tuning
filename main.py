# %% Imports
import random
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# %% Read in dataset.
data = pd.read_csv("impstroke.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)

# %% Split data from outputs
X = data.drop("stroke", axis=1)
y = data["stroke"]

# %% Oversample data
x_resampled, y_resampled = RandomOverSampler(
    sampling_strategy=0.25, random_state=11).fit_resample(X, y)

# %% Test Train Split
# X_train, X_test, y_train, y_test = train_test_split(
#     x_resampled, y_resampled, test_size=0.20, random_state=97)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=97)

# %%
model = xgb.XGBClassifier(verbosity=0, use_label_encoder=False)
model.fit(X_train, y_train)
print("f1_score:", f1_score(y_test, model.predict(X_test)))
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# %% Define Genetic Algorithm Functions

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


def fitness_score(y_true, y_pred, type):
    if type == 'acc':
        return round((accuracy_score(y_true, y_pred)), 4)
    if type == 'f1':
        return round((f1_score(y_true, y_pred)), 4)
    if type == 'bal_acc':
        return round((balanced_accuracy_score(y_true, y_pred)), 4)
    else:
        return NotImplementedError


def train_population(population, dMatrixTrain, dMatrixtest, y_test):
    aScore = []
    f1Score = []
    balAccScore = []
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
        num_round = 5
        xgbT = xgb.train(param, dMatrixTrain, num_round)
        preds = xgbT.predict(dMatrixtest)
        preds = preds > 0.5
        aScore.append(fitness_score(y_test, preds, 'acc'))
        f1Score.append(fitness_score(y_test, preds, 'f1'))
        balAccScore.append(fitness_score(y_test, preds, 'bal_acc'))
    return [aScore, f1Score, balAccScore]


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


def mutation(crossover, selectedParentsStats, mu=0.25):
    mutation = crossover.tolist()  # The array of array of hyper-parameters
    # The total length of the hyper-parameter array is the numberOfParameters

    counterLoopMutation = 0
    for i in mutation:
        percentageCheck = round(random.random(), 3)

        if percentageCheck < mu:
            mutation.pop(counterLoopMutation)
            newHyperparameterArray = []

            # Hyper-parameter 0
            newParameter0 = round(np.random.normal(
                selectedParentsStats['mean'][0], selectedParentsStats['sd'][0]), 2)
            if newParameter0 < 0.01:
                newHyperparameterArray.append(round(0.01, 2))
            if newParameter0 > 1.00:
                newHyperparameterArray.append(round(1.00, 2))
            if newParameter0 <= 1.00 and newParameter0 >= 0.01:
                newHyperparameterArray.append(newParameter0)

            # Hyper-parameter 1
            newParameter1 = round(np.random.normal(
                selectedParentsStats['mean'][1], selectedParentsStats['sd'][1]), 0)
            if newParameter1 < 10.00:
                newHyperparameterArray.append(round(10.00, 0))
            if newParameter1 > 1500:
                newHyperparameterArray.append(round(1500, 0))
            if newParameter1 <= 1500 and newParameter1 >= 10.00:
                newHyperparameterArray.append(newParameter1)

            # Hyper-parameter 2
            newParameter2 = round(np.random.normal(
                selectedParentsStats['mean'][2], selectedParentsStats['sd'][2]), 0)
            if newParameter2 < 1.00:
                newHyperparameterArray.append(round(1.0, 0))
            if newParameter2 > 10.00:
                newHyperparameterArray.append(round(10.0, 0))
            if newParameter2 <= 10.00 and newParameter2 >= 1.00:
                newHyperparameterArray.append(newParameter2)

            # Hyper-parameter 3
            newParameter3 = round(np.random.normal(
                selectedParentsStats['mean'][3], selectedParentsStats['sd'][3]), 2)
            if newParameter3 < 0.01:
                newHyperparameterArray.append(round(.01, 2))
            if newParameter3 > 10.00:
                newHyperparameterArray.append(round(10.00, 2))
            if (newParameter3 >= 0.01) and (newParameter3 <= 10.00):
                newHyperparameterArray.append(newParameter3)

            # Hyper-parameter 4
            newParameter4 = round(np.random.normal(
                selectedParentsStats['mean'][4], selectedParentsStats['sd'][4]), 2)
            if newParameter4 < 0.01:
                newHyperparameterArray.append(round(.01, 2))
            if newParameter4 > 10.00:
                newHyperparameterArray.append(round(10.00, 2))
            if (newParameter4 >= 0.01) and (newParameter4 <= 10.00):
                newHyperparameterArray.append(newParameter4)

            # Hyper-parameter 5
            newParameter5 = round(np.random.normal(
                selectedParentsStats['mean'][5], selectedParentsStats['sd'][5]), 2)
            if newParameter5 < 0.01:
                newHyperparameterArray.append(round(.01, 2))
            if newParameter5 > 1.00:
                newHyperparameterArray.append(round(1.00, 2))
            if (newParameter5 >= 0.01) and (newParameter5 <= 1.00):
                newHyperparameterArray.append(newParameter5)

            # Hyper-parameter 6
            newParameter6 = round(np.random.normal(
                selectedParentsStats['mean'][6], selectedParentsStats['sd'][6]), 2)
            if newParameter6 < 0.01:
                newHyperparameterArray.append(round(.01, 2))
            if newParameter6 > 1.00:
                newHyperparameterArray.append(round(1.00, 2))
            if (newParameter6 >= 0.01) and (newParameter6 <= 1.00):
                newHyperparameterArray.append(newParameter6)

            mutation.append(newHyperparameterArray)

        counterLoopMutation = counterLoopMutation + 1

    mutation2 = []
    for i in range(len(mutation)):
        mutation2.append(np.array(mutation[i]))

    mutation3 = np.array(mutation2)
    return mutation3

# %% Basline mutation model
def mutation_base(crossover, selectedParentsStat, mu=0.25):
    mutation = crossover.tolist() #The array of array of hyper-parameters
    #The total length of the hyper-parameter array is the numberOfParameters

    counterLoopMutation = 0
    for i in mutation:
        percentageCheck = round(random.random(),3)
        
        if percentageCheck < mu:
            mutation.pop(counterLoopMutation)
            newHyperparameterArray = []
            
            #Hyper-parameter 0
            newParameter0 = round(random.uniform(0.01, 1), 2)
            newHyperparameterArray.append(newParameter0)

            #Hyper-parameter 1
            newParameter1 = random.randrange(10, 1500, step=25)
            newHyperparameterArray.append(newParameter1)
            
            #Hyper-parameter 2
            newParameter2 = int(random.randrange(1, 10, step=1))
            newHyperparameterArray.append(newParameter2)

            #Hyper-parameter 3
            newParameter3 = round(random.uniform(0.01, 10.0), 2)
            newHyperparameterArray.append(newParameter3)
            
            #Hyper-parameter 4
            newParameter4 = round(random.uniform(0.01, 10.0), 2)
            newHyperparameterArray.append(newParameter4)
            
            #Hyper-parameter 5
            newParameter5 = round(random.uniform(0.01, 1.0), 2)
            newHyperparameterArray.append(newParameter5)
            
            #Hyper-parameter 6
            newParameter6 = round(random.uniform(0.01, 1.0), 2)
            newHyperparameterArray.append(newParameter6)
            
            mutation.append(newHyperparameterArray)
 

        counterLoopMutation = counterLoopMutation + 1
    
    mutation2 = []
    for i in range(len(mutation)): 
        mutation2.append(np.array(mutation[i]))
    
    mutation3 = np.array(mutation2)

    return mutation3

# %% Main GA Loop

def start_ga(n_parents=64, n_parents_mating=32, n_params=7, n_gens=5, mu_func=mutation, mu=0.25):
    xgDMatrix = xgb.DMatrix(X_train, y_train)
    xgbDMatrixTest = xgb.DMatrix(X_test, y_test)

    populationSize = (n_parents, n_params)
    population = initilialize_poplulation(n_parents)
    # fitnessHistory = np.empty([numberOfGenerations+1, 2, numberOfParents])
    fitnessHistory = np.empty([n_gens, 3, n_parents])
    populationHistory = np.empty(
        [(n_gens+1)*n_parents, n_params])
    populationHistory[0:n_parents, :] = population

    for generation in range(n_gens):
        print("This is number %s generation" % (generation))

        fitness_vals = train_population(
            population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
        fitnessHistory[generation, 0, :] = fitness_vals[0]  # Accuracy
        fitnessHistory[generation, 1, :] = fitness_vals[1]  # F1 Score
        fitnessHistory[generation, 2, :] = fitness_vals[2]  # F1 Score

        print('Best Acc score in this generation = {}'.format(
            np.max(fitnessHistory[generation, 0, :])))
        print('Best F1 score in this generation = {}'.format(
            np.max(fitnessHistory[generation, 1, :])))
        print('Best Bal Acc score in this generation = {}'.format(
            np.max(fitnessHistory[generation, 2, :])))

        parents = new_parents_selection(
            population=population, fitness=fitness_vals[1], numParents=n_parents_mating)

        feature_statistics = {
            'mean': np.mean(parents, axis=0),
            'sd': np.std(parents, axis=0)
        }

        children = crossover_uniform(parents=parents, childrenSize=(
            populationSize[0] - parents.shape[0], n_params))
        children_mutated = mu_func(
            children, feature_statistics, mu)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = children_mutated
        populationHistory[(generation+1)*n_parents: (generation+1)
                          * n_parents + n_parents, :] = population

        print("------")

    # fitness_vals = train_population(
    #     population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest, y_test=y_test)
    # fitnessHistory[generation+1, 0, :] = fitness_vals[0] # Accuracy
    # fitnessHistory[generation+1, 1, :] = fitness_vals[1] # F1 Score
    bestFitnessIndex = np.where(
        fitness_vals[1] == np.max(fitness_vals[1]))[0][0]

    print("Best fitness is =", fitness_vals[1][bestFitnessIndex])
    print("Best parameters are:")
    print('learning_rate', population[bestFitnessIndex][0])
    print('n_estimators', population[bestFitnessIndex][1])
    print('max_depth', int(population[bestFitnessIndex][2]))
    print('min_child_weight', population[bestFitnessIndex][3])
    print('gamma', population[bestFitnessIndex][4])
    print('subsample', population[bestFitnessIndex][5])
    print('colsample_bytree', population[bestFitnessIndex][6])

    max_fit_scores = np.max(fitnessHistory, axis=2)
    max_fit_scores = np.transpose(max_fit_scores)

    # %% Visualize Accuracy Improvement over Generations
    plt.plot(max_fit_scores[0], label="Acc", scalex=1)
    plt.title("Accuracy v. Generation")
    plt.xlabel("Generation")
    plt.ylabel("Acc")
    plt.xticks(range(len(max_fit_scores[0])))
    if in_notebook():
        plt.show()
    plt.savefig('figures\\' + '{:0>2}'.format(mu) + '_acc.png')
    plt.clf()

    # %% Visualize F1 Score Improvement over Generations
    plt.plot(max_fit_scores[1], label="F1 Score")
    plt.title("F1 Score v. Generation")
    plt.xlabel("Generation")
    plt.ylabel("F1")
    plt.xticks(range(len(max_fit_scores[0])))
    if in_notebook():
        plt.show()
    plt.savefig('figures\\' + '{:0>2}'.format(mu) + '_f1.png')
    plt.clf()


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except:
        return False
    return True


# %% Re-run GA with multiple mutation rates
# for mu in [x / 100.0 for x in range(5, 50, 5)]:
#     print("-----------------------------------------")
#     print("MUTATION RATE IS ", '{:0>2}'.format(mu))
#     print("------")
#     start_ga(n_parents=64, n_parents_mating=32, n_params=7, n_gens=25, mu=mu)
#     print("\n\n")
n_gens=5
print("OUR MUTATION FUNC")
start_ga(n_parents=64, n_parents_mating=32, n_params=7, n_gens=n_gens, mu=0.25)
print("BASELINE MUTATION FUNC")
start_ga(n_parents=64, n_parents_mating=32, n_params=7, n_gens=n_gens, mu=0.25, mu_func=mutation_base)

# %%
