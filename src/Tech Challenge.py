import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Carregar dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função de avaliação da aptidão
def evaluate(individual):
    n_estimators = int(individual[0])
    max_depth = int(individual[1])
    min_samples_split = int(individual[2])

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,

# Configurar DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 10, 100)  # n_estimators
toolbox.register("attr_int2", np.random.randint, 1, 20)   # max_depth
toolbox.register("attr_int3", np.random.randint, 2, 10)   # min_samples_split

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int2, toolbox.attr_int3), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[10, 1, 2], up=[100, 20, 10], eta=0.1, indpb=0.1)


# Configurar o algoritmo genético
def main():
    population = toolbox.population(n=50)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)

if __name__ == "__main__":
    main()