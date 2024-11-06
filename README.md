# Hybird-NN-GA-method
Apply artifical neural network for fitness funtion approximation and use genetic algorithm to find the local optimal

Three layer MLP

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


<img width="800" alt="image" src="https://github.com/user-attachments/assets/d5f6a639-950a-4077-b392-7dc6bd4d35dc">

Feature importance analysis
<img width="296" alt="image" src="https://github.com/user-attachments/assets/d4588475-5c24-4e8e-a251-444cbc76f598">

