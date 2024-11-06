# Hybrid Neural Network - Genetic Algorithm Optimization
## Overview
This project implements a hybrid optimization approach combining Artificial Neural Networks (ANN) for fitness function approximation with Genetic Algorithms (GA) for finding local optima. The method leverages the prediction capabilities of neural networks while using evolutionary search to explore the solution space effectively.

## Features
- Three-layer Multilayer Perceptron (MLP) for function approximation
- Genetic Algorithm optimization with customizable parameters
- Feature importance analysis
- Visualization of search progress and results

Three layer MLP:

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output

Optimization search:
# Evolutionary algorithm
for gen in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace the old population with the offspring
    pop[:] = offspring

    # Update the hall of fame and the statistics with the currently evaluated population
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(evals=len(invalid_ind), gen=gen, **record)

    print(logbook.stream)

    # Save more data along the evolution for later plotting
    fbest[gen] = hof[0].fitness.values
    best[gen, :N] = hof[0]
    std[gen, :N] = np.std([ind for ind in pop], axis=0)

print("Best individual is:", hof[0], hof[0].fitness.values)


<img width="800" alt="image" src="https://github.com/user-attachments/assets/d5f6a639-950a-4077-b392-7dc6bd4d35dc">

Feature importance analysis
<img width="296" alt="image" src="https://github.com/user-attachments/assets/d4588475-5c24-4e8e-a251-444cbc76f598">

Example of how searching work
![AntColony](https://github.com/user-attachments/assets/6acf322f-1d27-4163-8ef0-620849d41c73)


Reference:
https://commons.wikimedia.org/wiki/File:AntColony.gif#/media/File:AntColony.gif
