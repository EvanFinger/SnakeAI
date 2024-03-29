import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, model_filename=None):
        
        super().__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        self.save(model_filename)
        
    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
    def save(self,  file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save(self, file_name)
            
            
class QTrainer:
    
    def __init__(self, model, learning_rate, gamma) -> None:
        
        self.learningRate = learning_rate
        self.gamma = gamma
        self.model = model
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        
        self.criterion = nn.MSELoss()
        
    def trainStep(self, state, action, reward, next_state, game_over):
        
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        if len(state.shape) == 1:
            # Only one number; want in form (1, x)
            # (batches, value)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)
            
        # 1: predicted Q values with current state
        prediction = self.model(state)
        
        target = prediction.clone()
        for index in range(len(game_over)):
            newQ = reward[index]
            if not game_over[index]:
                newQ = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
                
            target[index][torch.argmax(action).item()] = newQ

        # 2: new Q = reward + gamma * max(next predicted Q value)
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction) # target -> newQ, prediction = Q
        loss.backward()
        
        self.optimizer.step()