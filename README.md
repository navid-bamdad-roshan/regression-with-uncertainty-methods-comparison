# Comparison of Neural Networks Regression Methods With Uncertainty Estimation

This project compares neural network regression model approaches that also provide uncertainty estimation. It implements neural network models to predict concrete strength using PyTorch. The models are trained on the concrete strength dataset, utilizing Mean Squared Error (MSE) as the loss function in the baseine model and the Adam optimizer for training.

**For more details refer to the [`./main.ipynb`](./main.ipynb) file.**
**Medium post: https://medium.com/@nbamdadroshan/which-neural-net-knows-what-it-doesnt-know-a-study-on-uncertainty-estimation-in-regression-1901382dd9b0**

## Project Structure

```
regression-with-uncertainty-methods-comparison
├── data
│   └── (Concrete strength dataset files)
├── src
│   ├── dataset.py        # Contains the ConcreteDataset class for loading data
│   ├── evaluate.py       # Contains the evaluation and plotting functions
│   ├── model.py          # Defines the Regression models with uncertainty estimation
│   └── trainer.py        # Contains the trainer classes
├── main.ipynb            # Main file to run the code and see the results.
├── requirements.txt      # Lists necessary Python packages
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd regression-with-uncertainty-methods-comparison
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the models, run the `main.ipynb` file:

## Dataset

The dataset used for training is located in the `data/` directory. If not existed, will be dowloaded automatically. It should contain the necessary files in CSV that can be processed by the `ConcreteDataset` class.

## Model

The neural network models are defined in `src/model.py` and is designed to predict concrete strength based on input features and provide uncertainty estimation along the predicted target value. The architecture can be modified as needed.
