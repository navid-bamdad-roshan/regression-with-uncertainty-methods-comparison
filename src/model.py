import torch
from torch.distributions import Normal
 





class BaseModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.0):
        super().__init__()

        layers = [
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ReLU(),
        ]

        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(torch.nn.ReLU())
            if dropout_prob > 0.0:
                layers.append(torch.nn.Dropout(p=dropout_prob))
        
        if dropout_prob > 0.0:
            layers.pop(-1)  # Remove last dropout layer

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x





class BaselineModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.0):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.base(x)
        y = self.output_layer(x)
        return y
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred, None # No uncertainty estimation for baseline model



class MeanSTDModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.0):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_mean = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.output_log_std = torch.nn.Linear(hidden_sizes[-1], output_size)


    def forward(self, x):
        x = self.base(x)
        x_mean = self.output_mean(x)
        x_log_std = self.output_log_std(x)
        return x_mean, x_log_std
    
    def get_dist_obj(self, y_mean, y_log_std):
        y_std = torch.exp(y_log_std)
        return Normal(loc=y_mean, scale=y_std)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_mean, y_log_std = self.forward(x)
        return y_mean, torch.exp(y_log_std)


class MeanVarianceModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.0):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_mean = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.output_log_variance = torch.nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.base(x)
        x_mean = self.output_mean(x)
        x_log_variance = self.output_log_variance(x)
        return x_mean, x_log_variance
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_mean, y_log_variance = self.forward(x)
        return y_mean, torch.exp(y_log_variance)


class MonteCarloDropoutModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.3):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.base(x)
        y = self.output_layer(x)
        return y
    
    def predict(self, x, num_samples=100):
        self.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                y_pred = self.forward(x)
                preds.append(y_pred.unsqueeze(0))
        preds = torch.cat(preds, dim=0)  # Shape: (num_samples, batch_size, output_size)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        return mean_pred, std_pred



class MonteCarloDropoutMeanSTDModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.3):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_mean = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.output_log_std = torch.nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.base(x)
        x_mean = self.output_mean(x)
        x_log_std = self.output_log_std(x)
        return x_mean, x_log_std
    
    def get_dist_obj(self, y_mean, y_log_std):
        y_std = torch.exp(y_log_std)
        return Normal(loc=y_mean, scale=y_std)
    
    def predict(self, x, num_samples=10):
        self.train()  # Enable dropout
        pred_means = []
        pred_log_stds = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred_mean, pred_log_std = self.forward(x)
                pred_means.append(pred_mean.unsqueeze(0))
                pred_log_stds.append(pred_log_std.unsqueeze(0))
        pred_means = torch.cat(pred_means, dim=0)  # Shape: (num_samples, batch_size, output_size)
        pred_log_stds = torch.cat(pred_log_stds, dim=0)  # Shape: (num_samples, batch_size, output_size)
        mean_pred = pred_means.mean(dim=0)
        std_pred = torch.exp(pred_log_stds).mean(dim=0) #* pred_means.std(dim=0)
        # std_pred = pred_means.std(dim=0)
        return mean_pred, std_pred

    

class PPOModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64, 32, 32], dropout_prob=0.0):
        super().__init__()
        self.base = BaseModel(input_size, hidden_sizes, dropout_prob=dropout_prob)
        self.output_mean = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.output_log_std = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.output_value = torch.nn.Linear(hidden_sizes[-1], output_size)
        self.log_std_tanh = torch.nn.Tanh()

        self.log_std = torch.nn.Parameter(torch.zeros(output_size))  # Learnable log standard deviation



    def forward(self, x):
        x = self.base(x)
        x_mean = self.output_mean(x)
        x_log_std = self.log_std_tanh(self.output_log_std(x)) * 10.0  # Scale to a [-10, 10] range
        x_value = self.output_value(x)
        x_log_std = (self.log_std.expand_as(x_mean) + x_log_std)/2
        return x_mean, x_log_std, x_value
    
    def get_dist_obj(self, y_mean, y_log_std):
        y_std = torch.exp(y_log_std)
        return Normal(loc=y_mean, scale=y_std)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_mean, y_log_std, y_value = self.forward(x=x)
        return y_mean, torch.exp(y_log_std)
    














