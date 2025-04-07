# Hybrid Neural Network - Genetic Algorithm Optimization

## Overview
This project implements a hybrid optimization approach combining Artificial Neural Networks (ANN) for fitness function approximation with Genetic Algorithms (GA) for finding local optima. The method leverages the prediction capabilities of neural networks while using evolutionary search to explore the solution space effectively.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/12b43d5a-b3e8-44ab-b4d3-d2c1b5a6480c">


## Features
- Three-layer Multilayer Perceptron (MLP) for function approximation
- Genetic Algorithm optimization with customizable parameters
- Feature importance analysis
- Visualization of search progress and results

## Architecture

### Neural Network Structure
The implementation uses a three-layer MLP with the following architecture:
```python
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
```

### Genetic Algorithm Components
- Selection: Tournament selection for choosing parent solutions
- Crossover: Uniform crossover with 0.5 probability
- Mutation: Gaussian mutation with 0.2 probability
- Population replacement: Generational replacement strategy

## Usage

### Installation
```bash
git clone https://github.com/yourusername/hybrid-nn-ga.git
cd hybrid-nn-ga
pip install -r requirements.txt
```

### Running the Optimization
```bash
python generic_NN_GA.py
```

## Algorithm Flow
1. Initialize population randomly
2. Train neural network on initial samples
3. For each generation:
   - Select parents using tournament selection
   - Apply crossover and mutation operators
   - Evaluate offspring using trained neural network
   - Update population
   - Update hall of fame and statistics
   - Record best solutions and population statistics

## Implementation Details

### Key Parameters
- `NGEN`: Number of generations
- Population size: Configurable based on problem complexity
- Crossover rate: 0.5
- Mutation rate: 0.2

### Optimization Loop
```python
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

    # Update the hall of fame and the statistics
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(evals=len(invalid_ind), gen=gen, **record)
    print(logbook.stream)

    # Save more data along the evolution for later plotting
    fbest[gen] = hof[0].fitness.values
    best[gen, :N] = hof[0]
    std[gen, :N] = np.std([ind for ind in pop], axis=0)

print("Best individual is:", hof[0], hof[0].fitness.values)
```

## Results

### Search Progress
The algorithm provides visualizations of convergence plots, population diversity, and best solution evolution.

### Feature Importance
The system includes feature importance analysis to identify key variables in the optimization process.

### Search Behavior
The search pattern resembles ant colony optimization, exploring multiple paths to find optimal solutions.

![AntColony](https://github.com/user-attachments/assets/6fffb934-de51-4747-9980-484855d3993c)


### Example optimized results
<img src="https://github.com/user-attachments/assets/181aa66e-c1e4-46be-a7dc-d030890aa17c" width="400" alt="Image 1">
<img src="https://github.com/user-attachments/assets/b4a23640-4011-4d06-b81e-0b1a21b2ca7d" width="400" alt="Image 2">
<img src="https://github.com/user-attachments/assets/598e82cc-316b-46e1-ad67-334282244c03" width="400" alt="Image 3">
<img src="https://github.com/user-attachments/assets/8ef4dc34-bc13-489d-96d6-b99675780fc0" width="400" alt="Image 4">
<img src="https://github.com/user-attachments/assets/ba3a874f-eba8-474b-94fd-a3c85f271411" width="400" alt="Image 5">



## Dependencies
- PyTorch
- DEAP
- NumPy
- Matplotlib

## References
- Base ant colony visualization: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AntColony.gif)
- DEAP framework documentation
- PyTorch neural network documentation
