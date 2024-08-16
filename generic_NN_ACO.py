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
# 2. COnvergence critieria


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
# num_epochs = 100
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
#Prediction part

# Define the predict function
def predict(model, input_parameters, sc):
    input_data = np.array(input_parameters).reshape(1, -1)
    scaled_input_data = sc.transform(input_data)
    input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
    return output

# Load the saved model state dictionary
model_state = torch.load('trained_model.pth')

# Create a new instance of your Net model
model = Net(input_size=X_train.shape[1], hidden_size1=64, hidden_size2=128, output_size=1)

# Load the saved state dictionary into the new model instance
model.load_state_dict(model_state)

# Define the fitness function
def evaluate(individual, model, fixed_inputs):
    inputs = np.concatenate((fixed_inputs, individual))
    output = predict(model, inputs, sc)
    fitness = -output.item()  # Negate the output to maximize it
    return fitness

# Define the fixed inputs
fixed_inputs = [2411, 6264, 6]

# Define the parameter bounds
p_LowerMass = (124.45, 150.65)
p_Area_a = (0.03, 0.08)
p_Area_H = (0.041, 0.066)
p_h = (1.72, 2.5)
p_Area_O = (0.0004, 0.0007)
p_DischargeC = (0.6, 0.95)

bounds = [p_LowerMass, p_Area_a, p_Area_H, p_h, p_Area_O, p_DischargeC]

# Convert bounds to separate lower and upper bounds for ACO
lb, ub = zip(*bounds)

# Define the optimization function for ACO
def aco_evaluate(individual):
    return evaluate(individual, model, fixed_inputs)

# ACO parameters
num_ants = 100
num_iterations = 400
evaporation_rate = 0.5
pheromone_deposit = 1.0
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of heuristic

# Initialize pheromone levels
dim = len(lb)
pheromone = np.ones((dim, 2))

# Heuristic information (can be problem-specific; here we use 1.0 for simplicity)
heuristic = np.ones((dim, 2))

# List to store the best fitness values and solutions at each iteration
fitness_values = []
best_solutions = []

# ACO algorithm
for iteration in range(num_iterations):
    solutions = []
    fitnesses = []

    for ant in range(num_ants):
        solution = []
        for i in range(dim):
            prob = pheromone[i] ** alpha * heuristic[i] ** beta
            prob /= prob.sum()
            value = np.random.choice([lb[i], ub[i]], p=prob)
            solution.append(value)
        
        fitness = aco_evaluate(solution)
        solutions.append(solution)
        fitnesses.append(fitness)

    # Find the best solution in this iteration
    best_index = np.argmin(fitnesses)
    best_solution = solutions[best_index]
    best_fitness = fitnesses[best_index]

    # Update pheromones
    for i in range(dim):
        if best_solution[i] == lb[i]:
            pheromone[i, 0] += pheromone_deposit
        else:
            pheromone[i, 1] += pheromone_deposit

    pheromone *= (1.0 - evaporation_rate)

    # Store the best fitness value and solution
    fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    print(f"Iteration {iteration+1}/{num_iterations}, best fitness: {best_fitness}")

# Get the best overall solution and fitness
best_overall_index = np.argmin(fitness_values)
best_fitness = fitness_values[best_overall_index]
best_individual = best_solutions[best_overall_index]

print("Best individual is:", best_individual, "with fitness:", best_fitness)

# Plot the fitness values against iterations
plt.plot(fitness_values)
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.title('Fitness Value vs. Iteration')
plt.grid(True)
plt.show()

# Save the fitness values to an Excel file
df = pd.DataFrame({'Iteration': np.arange(1, num_iterations + 1), 'Fitness Value': fitness_values})
df.to_excel('ACO_fitness_values_aco.xlsx', index=False)