import torch
import torch.nn as nn
import torch 
import numpy as np
from deap import base, creator, tools, algorithms
import random
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from deap import cma

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Suppress warnings
warnings.filterwarnings('ignore', module='sklearn')

#TODO
# 1. Make a multi object optimization 
# 2. Convergence critieria


os.chdir(r'D:\Machine learning\LandAbsorber')
# Read the CSV file
data = pd.read_csv("workingCondition_2.csv")

# Print the length of the dataframe
print(f'original data length: {len(data)}')


#print(data.head())
req_col_names = ["UpperMass", "Pressure_a0", "z_p0", "LowerMass", "Area_a",
                 "Area_H", "h", "Area_O", "DischargeC", "Max_F_oil", "Max_F_air", "Cavitation","Efficiency"]
curr_col_names = list(data.columns)
mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)

# Drop rows where 'Efficiency' column has NaN values (in-place)
data.dropna(subset=['Efficiency'], inplace=True)

# Print the length of the dataframe after filtering NaN values
print(f'After filtered data length: {len(data)}')

# Filter out rows where 'Max_F_oil' is higher than 60000
data = data[data['Max_F_oil'] <= 40000]
data = data[data['Max_F_air'] <= 50000]
data = data[data['Efficiency'] <= 0.8]
data = data[data['Cavitation'] <= 100]
# Print the length of the dataframe after filtering 'Max_F_oil' values
print(f'Final filtered data length: {len(data)}')


# Create a custom dataset class
class LandTwoDOFDataset(Dataset):
    def __init__(self, X, y1):
        #self.X = torch.from_numpy(X.values).float()  # Convert DataFrame to NumPy array
        self.X = torch.from_numpy(X).float()
        self.y1 = torch.from_numpy(y1.values.reshape(-1, 1)).float()

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, index):
        return self.X[index], self.y1[index]
    
# Define the neural network architecture
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

X = data.iloc[:,:-4]         # Features Exclude, hydraulic, pneumaic, cavitation and effieciency
y1 = data.iloc[:,-1]          # Target - the last second Column

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=2)

#scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Create the dataset and dataloader
train_dataset = LandTwoDOFDataset(X_train, y1_train)
test_dataset = LandTwoDOFDataset(X_test, y1_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define the model, loss function, and optimizer
model = Net(input_size=X_train.shape[1], hidden_size1=64, hidden_size2=128, output_size=1)
criterion1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# # Train the model
# num_epochs = 10
# train_losses1 = []
# epoch_target1 = []
# epoch_true1 = []
# true_values1 = []
# target_values1 = []

# for epoch in range(num_epochs):
#     epoch_losses1 = []
#     epoch_target1 = []
#     epoch_true1 = []
#     for X, y1 in train_loader:
#         optimizer.zero_grad()
#         outputs1= model(X)
#         loss = criterion1(outputs1, y1)
#         loss.backward()
#         optimizer.step()
#         epoch_losses1.append(loss.item())

#         # Append true and target values for output 1
#         epoch_target1.append(torch.mean(y1).item())
#         epoch_true1.append(torch.mean(outputs1).item())

#     epoch_loss1 = np.mean(epoch_losses1)
#     train_losses1.append(epoch_loss1)

#     epoch_target1 = np.mean(epoch_target1)
#     target_values1.append(epoch_target1)

#     epoch_true1 = np.mean(epoch_true1)
#     true_values1.append(epoch_true1)

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss1:.4f}')

# # Assuming 'model' is your trained PyTorch model
# # Save the trained model
# torch.save(model.state_dict(), 'trained_model.pth')

#Prediction part

def predict(model, input_parameters, sc):
    """
    Function to perform prediction using the given model, input parameters, and scaler.
    
    Args:
    - model: PyTorch model for prediction
    - input_parameters: List or array of input parameters
    - sc: Scaler object for transforming input data
    
    Returns:
    - output: Model's prediction
    """
    # Convert input parameters to numpy array and reshape
    input_data = np.array(input_parameters).reshape(1, -1)
    
    # Scale input data
    scaled_input_data = sc.transform(input_data)
    
    # Convert scaled input data to torch tensor
    input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32)
    
    # Perform prediction using the loaded model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(input_tensor)
    return output


# Load the saved model state dictionary
model_state = torch.load('trained_model.pth')

# Create a new instance of your Net model
model = Net(input_size=X_train.shape[1], hidden_size1=64, hidden_size2=128, output_size=1)

# Load the saved state dictionary into the new model instance
model.load_state_dict(model_state)


#Define the fitness function


# Define the problem
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define the ranges for each parameter
p_LowerMass= (124.45, 150.65)
p_Area_a = (0.03, 0.08)
p_Area_H = (0.041, 0.066)
p_h = (1.72, 2.5)
p_Area_O = (0.0004, 0.0007)
p_DischargeC = (0.6, 0.95)

# Register the generation functions
toolbox.register("attr_float_4", random.uniform, *p_LowerMass)
toolbox.register("attr_float_5", random.uniform, *p_Area_a)
toolbox.register("attr_float_6", random.uniform, *p_Area_H)
toolbox.register("attr_float_7", random.uniform, *p_h)
toolbox.register("attr_float_8", random.uniform, *p_Area_O)
toolbox.register("attr_float_9", random.uniform, *p_DischargeC)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_4, toolbox.attr_float_5, toolbox.attr_float_6, toolbox.attr_float_7, toolbox.attr_float_8, toolbox.attr_float_9), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Set the fixed inputs
fixed_inputs = [2411, 6264, 6]

# Define the evaluation function
def evaluate(individual, model, fixed_inputs):
    inputs = np.concatenate((fixed_inputs, individual))
    output = predict(model, inputs, sc)
    fitness = -output.item()  # Negate the output to maximize it
    return fitness,

# Register the evaluate function
toolbox.register("evaluate", evaluate, model=model, fixed_inputs=fixed_inputs)

# Register the genetic operators
toolbox.register("mate", tools.cxTwoPoint)

def mutate_with_range(individual, ranges, indpb):
    for i, value in enumerate(individual):
        if random.random() < indpb:
            individual[i] = ranges[i][0] + (ranges[i][1] - ranges[i][0]) * random.random()
    return individual,

RANGES = [p_LowerMass, p_Area_a, p_Area_H, p_h, p_Area_O, p_DischargeC]
toolbox.register("mutate", mutate_with_range, ranges=RANGES, indpb=0.1)

toolbox.register("select", tools.selTournament, tournsize=3)  # Increase selection pressure

# Number of generations
NGEN = 200
N = 6

# Initialize arrays
sigma = np.ndarray((NGEN, 1))
axis_ratio = np.ndarray((NGEN, 1))
diagD = np.ndarray((NGEN, N))
fbest = np.ndarray((NGEN, 1))
best = np.ndarray((NGEN, N))
std = np.ndarray((NGEN, N))

# Initialize the population
np.random.seed(64)
pop = toolbox.population(n=2000)  # Initialize population outside of the loop
hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

# Evaluate the initial population
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

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

# # Plotting the standard deviation over generations
# plt.figure(figsize=(10, 5))
# plt.plot(range(NGEN), std.mean(axis=1), label='Standard Deviation')
# plt.xlabel('Generation')
# plt.ylabel('Standard Deviation')
# plt.title('Standard Deviation of the Population over Generations')
# plt.legend()
# plt.grid(True)
# plt.show()


param_ranges = [
    p_LowerMass[1] - p_LowerMass[0],
    p_Area_a[1] - p_Area_a[0],
    p_Area_H[1] - p_Area_H[0],
    p_h[1] - p_h[0],
    p_Area_O[1] - p_Area_O[0],
    p_DischargeC[1] - p_DischargeC[0]
]

# Collect data for plotting
#x = list(range(0, len(pop) * NGEN, len(pop)))
x = range(NGEN)
avg, max_, min_ = logbook.select("avg", "max", "min")

# Function to compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Calculate moving average standard deviations with window size of 10
window_size = 10
std_avg = np.array([moving_average(std[:, i], window_size) for i in range(N)]).T

# Adjust x values for the moving average plot
x_avg = x[(window_size - 1):]

# Define marker symbols for each parameter
markers = ['o', 's', '^', 'D', 'v', 'P']

# Plot Average Standard Deviations in All Coordinates
fig, ax = plt.subplots(figsize=(10, 10))
parameter_names = ["m₂", "Aₐ", "Aₕ", "h", "A₀", "Dc"]
for i in range(N):
    ax.semilogy(x_avg, std_avg[:, i] / param_ranges[i], label=parameter_names[i], linestyle='-', marker=markers[i], markersize=5)


#ax.set_title("Average Standard Deviations in All Coordinates", fontsize=14)
ax.legend(loc='upper right', fontsize=28, frameon=False)
#ax.grid(True)
# Set x-axis limit from 0 to the maximum value
ax.set_xlim(0, 400)
# Increase font size of x and y-axis tick labels
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)

# Increase thickness of axis lines
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)


# Increase font size of x and y-axis labels
ax.set_xlabel("Generation", fontsize=28)
ax.set_ylabel("Standard Deviations", fontsize=28)
plt.savefig('plot.png', dpi=600)
plt.show()

# Plot Average, Max, and Min Fitness
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, avg, "--b", label='Average Fitness', linewidth=2)
ax.plot(x, max_, "--r", label='Max Fitness', linewidth=2)
ax.plot(x, min_, "-g", label='Min Fitness', linewidth=2)
#ax.set_title("Average, Max, and Min Fitness", fontsize=14)
ax.legend(fontsize=28)

plt.show()

# Create a dictionary with the best individual's parameters and fitness value
data = {
    'Parameter': ['Fixed_UpperMass', 'Fixed_Pressure_a0', 'Fixed_z_p0', 'LowerMass', 'Area_a', 'Area_H', 'h', 'Area_O', 'DischargeC', 'Fitness'],
    'Value': fixed_inputs + list(hof[0]) + list(hof[0].fitness.values)
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
df.to_excel('optimization_GA_160.xlsx', index=False)

print("Best individual and its fitness have been stored to optimization_GA.xlsx")


print("\nTop ten random searching results of the hybrid GA-NN method:")
print('Fixed_UpperMass', 'Fixed_Pressure_a0', 'Fixed_z_p0', 'LowerMass', 'Area_a', 'Area_H', 'h', 'Area_O', 'DischargeC', 'Fitness')


# Make the Individual class hashable
def hashIndividual(individual):
    """Compute a hash value for the individual"""
    return hash(tuple(individual))

creator.Individual.__hash__ = hashIndividual

def eqIndividuals(ind1, ind2):
    """Compare two individuals for equality"""
    return tuple(ind1) == tuple(ind2)

creator.Individual.__eq__ = eqIndividuals

# Select the top 10 individuals
top_individuals = tools.selBest(set(pop), k=10)

for ind in top_individuals:
    print(ind, ind.fitness.values[0])




# # The x-axis will be the number of evaluations
# x = list(range(0, len(population) * NGEN, len(population)))
# avg, max_, min_ = logbook.select("avg", "max", "min")

# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# for i in range(N):
#     plt.semilogy(x, std[:, i] / param_ranges[i], label=parameter_names[i], linestyle='-', marker='o', markersize=3)
# #plt.semilogy(x, std)
# plt.title("Standard Deviations in All Coordinates")
# #plt.legend(loc='center left', bbox_to_anchor=(-0.35, 0.5))  # Moving legend outside to the left
# plt.legend(loc='upper right')  # Adding legend in the upper right corner

# plt.subplot(2, 2, 2)
# plt.plot(x, fbest, label='Best Fitness')
# plt.title("Object Variables")
# plt.legend()  # Adding legend

# plt.subplot(2, 2, 3)
# plt.plot(x, avg, "--b", label='Average Fitness')
# plt.plot(x, max_, "--r", label='Max Fitness')
# plt.plot(x, min_, "-g", label='Min Fitness')
# plt.title("Average, Max, and Min Fitness")
# plt.legend()  # Adding legend

# plt.tight_layout()
# plt.show()



