import torch
from src.dataset import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal




class BaselineModelTrainer:
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        # Prepare data loader
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Move model to the specified device
        self.model.to(self.device)

        # Training loop
        progress_bar = tqdm(range(self.num_epochs), desc='Training Progress', leave=True)
        self.model.train()
        for epoch in progress_bar:
            running_loss = 0.0

            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                running_loss += self.forward_and_backward(inputs, targets)

            # Print epoch loss
            epoch_loss = running_loss / len(data_loader)
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=epoch_loss)
    
    def forward_and_backward(self, inputs, targets):

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        # loss = self.train_obj_dict['criterion'](outputs, targets)
        loss = ((outputs - targets) ** 2).mean() 

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()


class MeanSTDModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward_and_backward(self, inputs, targets):

        # Forward pass
        y_mean, y_log_std = self.model(inputs)
        dist_obj = self.model.get_dist_obj(y_mean, y_log_std)

        # Compute negative log likelihood loss
        nll_loss = -dist_obj.log_prob(targets).mean()

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss.item()


class MeanVarianceModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward_and_backward(self, inputs, targets):

        # Forward pass
        y_mean, y_log_variance = self.model(inputs)
        # dist_obj = self.model.get_dist_obj(y_mean, y_log_std)

        # Compute negative log likelihood loss
        nll_loss = (0.5 * y_log_variance) + (((targets - y_mean) ** 2) / (2 * torch.exp(y_log_variance)))
        nll_loss = nll_loss.mean()

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss.item()


class MonteCarloDropoutModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, mc_samples=10, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.mc_samples = mc_samples
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward_and_backward(self, inputs, targets):

        # Forward pass with MC Dropout
        preds = []
        for _ in range(self.mc_samples):
            preds.append(self.model(inputs))
        preds = torch.stack(preds)  # Shape: (mc_samples, batch_size, output_size)
        pred_mean = preds.mean(dim=0)

        # Compute loss
        loss = ((pred_mean - targets) ** 2).mean()

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()


class MonteCarloDropoutMeanSTDModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, mc_samples=10, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.mc_samples = mc_samples
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward_and_backward(self, inputs, targets):

        # Forward pass with MC Dropout
        pred_mean = []
        pred_log_std = []
        for _ in range(self.mc_samples):
            mean, log_std = self.model(inputs)
            pred_mean.append(mean)
            pred_log_std.append(log_std)
        pred_mean = torch.stack(pred_mean)  # Shape: (mc_samples, batch_size, output_size)
        pred_log_std = torch.stack(pred_log_std)  # Shape: (mc_samples, batch_size, output_size)
        pred_mean = pred_mean.mean(dim=0)
        pred_log_std = pred_log_std.mean(dim=0)

        dist_obj = self.model.get_dist_obj(pred_mean, pred_log_std)

        # Compute negative log likelihood loss
        nll_loss = -dist_obj.log_prob(targets).mean()

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss.item()



class MonteCarloDropoutMeanVarianceModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, mc_samples=10, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.mc_samples = mc_samples
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward_and_backward(self, inputs, targets):

        # Forward pass with MC Dropout
        pred_mean = []
        pred_log_vaiance = []
        for _ in range(self.mc_samples):
            mean, log_vaiance = self.model(inputs)
            pred_mean.append(mean)
            pred_log_vaiance.append(log_vaiance)
        pred_mean = torch.stack(pred_mean)  # Shape: (mc_samples, batch_size, output_size)
        pred_log_vaiance = torch.stack(pred_log_vaiance)  # Shape: (mc_samples, batch_size, output_size)
        pred_mean = pred_mean.mean(dim=0)
        pred_log_vaiance = pred_log_vaiance.mean(dim=0)

        # Compute negative log likelihood loss
        nll_loss = (0.5 * pred_log_vaiance) + (((targets - pred_mean) ** 2) / (2 * torch.exp(pred_log_vaiance)))
        nll_loss = nll_loss.mean()

        # Zero the gradients
        self.optimizer.zero_grad()
        # Backward pass and optimization
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss.item()


class PPOModelTrainer(BaselineModelTrainer):
    def __init__(self, model, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, sample_size=10, learn_steps=4, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.sample_size = sample_size
        self.learn_steps = learn_steps
    
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        # Prepare data loader
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Move model to the specified device
        self.model.to(self.device)

        # Training loop
        progress_bar = tqdm(range(self.num_epochs), desc='Training Progress', leave=True)
        self.model.train()
        for epoch in progress_bar:
            running_loss = 0.0
            running_reward = 0.0

            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_loss, batch_reward = self.forward_and_backward(inputs, targets)
                running_loss += batch_loss
                running_reward += batch_reward

            
            epoch_loss = running_loss / len(data_loader)
            epoch_reward = running_reward / len(data_loader)
            # Update the progress bar with the current loss
            progress_bar.set_postfix(reward=epoch_reward)

    def forward_and_backward(self, inputs, targets):

        # Forward pass for sampling
        with torch.no_grad():
            y_mean, y_log_std, y_value = self.model(inputs)
            dist_obj = self.model.get_dist_obj(y_mean, y_log_std)
            sampled_preds = dist_obj.sample(sample_shape=torch.Size([self.sample_size])) # Shape: (sample_size, batch_size, output_size)
            old_log_probs = dist_obj.log_prob(sampled_preds).sum(dim=-1)  # Shape: (sample_size, batch_size)

            rewards = -((sampled_preds - targets.unsqueeze(0)) ** 2).mean(dim=-1)  # Shape: (sample_size, batch_size)
            
            # No need for GAE calculation to compute advantages since this prediction is completely independent to other predictions.
            # Advantages is just computed by subracting expected reward (value) from current received reward.
            # Positive and higher advantage means the received reward was better than the expected reward.
            # Negative and lower advantage means the received reward was worse than the expected reward.
            advantages = rewards - y_value.unsqueeze(0).squeeze(-1)  # Shape: (sample_size, batch_size)

            # normalizing the advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for i in range(self.learn_steps):
            
            # Forward pass to get current predictions
            y_mean_grad, y_log_std_grad, y_value_grad = self.model(inputs)
            dist_obj_grad = self.model.get_dist_obj(y_mean_grad, y_log_std_grad)

            new_log_probs = dist_obj_grad.log_prob(sampled_preds).sum(dim=-1)  # Shape: (sample_size, batch_size)

            prob_ratio = torch.exp(new_log_probs - old_log_probs)  # Shape: (sample_size, batch_size)

            value_loss = ((y_value_grad.squeeze(-1) - rewards.mean(dim=0)) ** 2).mean()

            entropy_loss = -dist_obj_grad.entropy().mean()
            

            # PPO clipped objective
            clip_epsilon = 0.2
            clipped_ratio = torch.clamp(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            ppo_objective = torch.min(prob_ratio * advantages, clipped_ratio * advantages)  # Shape: (sample_size, batch_size)
            ppo_loss = -ppo_objective.mean()

            # using a bigger entopy loss since the log_std of each instance is also 
            # predicted by model based on input features instead of a shared learnable parameter for all instances.
            loss = ppo_loss + 0.5 * value_loss + 0.9 * entropy_loss 

            # Zero the gradients
            self.optimizer.zero_grad()
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        return loss.item(), rewards.mean().item()