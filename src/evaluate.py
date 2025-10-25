import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, train_dataset, test_dataset, device='cpu'):
    """
    Evaluates the model on the training and testing sets, and plots the results.

    Args:
        model (torch.nn.Module): The trained model.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        test_dataset (torch.utils.data.Dataset): The testing dataset.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Get all features and labels
        X_train, y_train_true = train_dataset.features, train_dataset.labels
        X_test, y_test_true = test_dataset.features, test_dataset.labels

        # Move data to the specified device
        X_train = X_train.to(device)
        X_test = X_test.to(device)

        # Make trian predictions
        y_train_pred, y_train_uncertainty = model.predict(X_train)
        y_train_pred = y_train_pred.cpu().numpy()

        # Make test predictions
        y_test_pred, y_test_uncertainty = model.predict(X_test)
        y_test_pred = y_test_pred.cpu().numpy()

    # Move true labels to CPU for metrics and plotting
    y_train_true = y_train_true.cpu().numpy()
    y_test_true = y_test_true.cpu().numpy()

    # Calculate MSE
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    test_mse = mean_squared_error(y_test_true, y_test_pred)

    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # --- Plot 1: MSE Bar Plot ---
    plt.figure(figsize=(8, 5))
    plt.bar(['Train MSE', 'Test MSE'], [train_mse, test_mse], color=['blue', 'orange'])
    plt.ylabel('Mean Squared Error')
    plt.title('Train vs. Test MSE')
    plt.show()

    # --- Plot 2: True vs. Predicted Scatter Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Training set
    ax1.scatter(y_train_true, y_train_pred, alpha=0.5)
    ax1.plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Training Set: True vs. Predicted')
    ax1.grid(True)

    # Right subplot: Testing set
    ax2.scatter(y_test_true, y_test_pred, alpha=0.5, color='orange')
    ax2.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Testing Set: True vs. Predicted')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_models_mse(models, titles, train_dataset, test_dataset, device='cpu'):
    """
    Evaluate multiple models and plot train/test MSE side-by-side.

    Args:
        models (list): list of torch models (or objects with predict method).
        titles (list): list of strings for model names (same length as models).
        train_dataset, test_dataset: datasets with .features and .labels (tensors).
        device (str): 'cpu' or 'cuda'.
    """
    if len(models) != len(titles):
        raise ValueError("models and titles must have the same length")

    train_mses = []
    test_mses = []

    X_train = train_dataset.features.to(device)
    X_test = test_dataset.features.to(device)
    y_train_true = train_dataset.labels.cpu().numpy().ravel()
    y_test_true = test_dataset.labels.cpu().numpy().ravel()

    for m in models:
        m.to(device)
        m.eval()
        with torch.no_grad():
            
            # predict train
            pred_train, _ = m.predict(X_train)

            # predict test
            pred_test, _ = m.predict(X_test)

            # convert to numpy
            pred_train = pred_train.cpu().numpy().ravel()
            pred_test = pred_test.cpu().numpy().ravel()


        train_mse = mean_squared_error(y_train_true, pred_train)
        test_mse = mean_squared_error(y_test_true, pred_test)

        train_mses.append(train_mse)
        test_mses.append(test_mse)

    # Print summary
    for title, tr, te in zip(titles, train_mses, test_mses):
        print(f"{title}: Train MSE={tr:.6f}, Test MSE={te:.6f}")

    # Plot side-by-side bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(titles, train_mses, color='tab:blue')
    ax1.set_title("Train MSEs")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(titles, rotation=45, ha='right')

    ax2.bar(titles, test_mses, color='tab:orange')
    ax2.set_title("Test MSEs")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels(titles, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()



def evaluate_models_true_vs_pred(models, titles, train_dataset, test_dataset, device='cpu', cmap_name='tab10', markersize=20):
    """
    For each model, plot a row with:
      - left subplot: training set true_vs_pred
      - right subplot: testing set true_vs_pred

    Each model gets a distinct color; train and test for the same model share that color.

    Args:
        models (list): list of models (torch.nn.Module or objects with predict method)
        titles (list): list of model names (same length as models)
        train_dataset, test_dataset: datasets with .features and .labels (tensors)
        device (str): 'cpu' or 'cuda'
        cmap_name (str): matplotlib colormap name to pick colors from
        markersize (int): scatter marker size
    """
    if len(models) != len(titles):
        raise ValueError("models and titles must have the same length")

    n = len(models)
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(n)]

    X_train = train_dataset.features.to(device)
    X_test = test_dataset.features.to(device)
    y_train_true = train_dataset.labels.cpu().numpy().ravel()
    y_test_true = test_dataset.labels.cpu().numpy().ravel()

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * max(1, n)))
    axes = np.atleast_2d(axes)

    for i, (m, title) in enumerate(zip(models, titles)):
        color = colors[i]
        m.to(device)
        m.eval()
        with torch.no_grad():
            # predict train
            pred_train, uncertainty_train = m.predict(X_train)

            # predict test
            pred_test, uncertainty_test = m.predict(X_test)

            # convert to numpy
            pred_train = pred_train.cpu().numpy().ravel()
            pred_test = pred_test.cpu().numpy().ravel()

        ax_train = axes[i, 0]
        ax_test = axes[i, 1]

        # determine common axis limits for nicer comparison
        all_vals = np.concatenate([y_train_true, pred_train, y_test_true, pred_test])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (vmax - vmin) * 0.03 if vmax > vmin else 0.1
        lims = (vmin - pad, vmax + pad)

        ax_train.scatter(y_train_true, pred_train, color=color, alpha=0.7, s=markersize)
        ax_train.plot(lims, lims, 'k--', linewidth=1)
        ax_train.set_xlim(lims)
        ax_train.set_ylim(lims)
        ax_train.set_xlabel('True Values')
        ax_train.set_ylabel('Predicted Values')
        ax_train.set_title(f"{title} — Train")
        ax_train.grid(True)

        ax_test.scatter(y_test_true, pred_test, color=color, alpha=0.7, s=markersize)
        ax_test.plot(lims, lims, 'k--', linewidth=1)
        ax_test.set_xlim(lims)
        ax_test.set_ylim(lims)
        ax_test.set_xlabel('True Values')
        ax_test.set_ylabel('Predicted Values')
        ax_test.set_title(f"{title} — Test")
        ax_test.grid(True)

    plt.tight_layout()
    plt.show()








def plot_mse_vs_uncertainty_quantile(models, titles, train_dataset, test_dataset, device='cpu', cmap_name='tab10', quantiles=None):
    """
    For each model, compute MSE of predictions as a function of uncertainty quantile thresholds.

    Args:
        models (list): list of models (torch.nn.Module or objects with predict method)
        titles (list): list of model names (same length as models)
        train_dataset, test_dataset: dataset with .features and .labels (tensors)
        device (str): 'cpu' or 'cuda'
        cmap_name (str): matplotlib colormap name to pick colors from
        quantiles (list or np.ndarray): optional list of quantiles to evaluate (default np.linspace(0, 0.95, 20))
    """
    if len(models) != len(titles):
        raise ValueError("models and titles must have the same length")

    if quantiles is None:
        quantiles = np.linspace(0, 0.8, 20)

    cmap = plt.get_cmap(cmap_name)
    n = len(models)
    colors = [cmap(i % cmap.N) for i in range(n)]

    X_train = train_dataset.features.to(device)
    y_train_true = train_dataset.labels.cpu().numpy().ravel()
    X_test = test_dataset.features.to(device)
    y_test_true = test_dataset.labels.cpu().numpy().ravel()

    # plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # axes = np.atleast_2d(axes)

    for i, (m, title) in enumerate(zip(models, titles)):
        color = colors[i]
        m.to(device)
        m.eval()
        with torch.no_grad():

            # predict train
            pred_train, uncertainty_train = m.predict(X_train)

            # predict test
            pred_test, uncertainty_test = m.predict(X_test)

            # convert to numpy
            pred_train = pred_train.cpu().numpy().ravel()
            pred_test = pred_test.cpu().numpy().ravel()
            if not uncertainty_train is None:
                uncertainty_train = uncertainty_train.cpu().numpy().ravel()
                uncertainty_test = uncertainty_test.cpu().numpy().ravel()

        mse_values_train = []
        mse_values_test = []
        for q in quantiles:
            # threshold for uncertainty (keep lower uncertainty predictions)
            if uncertainty_train is None:
                mask_train = np.full(y_train_true.shape, fill_value=True)
            else:
                thresh_train = np.quantile(uncertainty_train, 1-q)
                mask_train = uncertainty_train <= thresh_train

            # if somehow mask is empty, skip
            if np.sum(mask_train) == 0:
                mse_values_train.append(np.nan)
            else:
                mse = mean_squared_error(y_train_true[mask_train], pred_train[mask_train])
                mse_values_train.append(mse)

            # threshold for uncertainty (keep lower uncertainty predictions)
            if uncertainty_test is None:
                mask_test = np.full(y_test_true.shape, fill_value=True)
            else:
                thresh_test = np.quantile(uncertainty_test, 1-q)
                mask_test = uncertainty_test <= thresh_test
                       
            # if somehow mask is empty, skip
            if np.sum(mask_test) == 0:
                mse_values_test.append(np.nan)
            else:
                mse = mean_squared_error(y_test_true[mask_test], pred_test[mask_test])
                mse_values_test.append(mse)

        axes[0].plot(quantiles, mse_values_train, label=title, color=color, marker='o', linewidth=2)
        axes[1].plot(quantiles, mse_values_test, label=title, color=color, marker='o', linewidth=2)
    

    axes[0].set_title("Train MSE vs. Uncertainty Quantile Threshold")
    axes[0].set_xlabel("Certainty Quantile Threshold (higher = more certain prediction)")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Test MSE vs. Uncertainty Quantile Threshold")
    axes[1].set_xlabel("Certainty Quantile Threshold (higher = more certain prediction)")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_uncertainty_grid(models, titles, test_dataset, device='cpu', quantiles=None, cmap_name='tab10', markersize=8):
    """
    Create a grid of scatter plots with shape (len(quantiles), len(models)).
    Rows = quantiles (i), Cols = models (j).
    For each cell (q, m) plot test instances where model uncertainty <= quantile threshold
    with alpha=0.7 and the other instances with alpha=0.3. X axis = true values, Y axis = predictions.

    Args:
        models (list): list of models (or objects with predict method returning (pred, uncertainty))
        titles (list): list of model names (same length as models)
        test_dataset: datasets having .features and .labels (tensors)
        device (str): 'cpu' or 'cuda'
        quantiles (array-like): quantile values in (0,1). If None defaults to np.linspace(0.1,0.9,5)
        cmap_name (str): matplotlib colormap name for assigning a unique color per model
        markersize (int): scatter marker size
    """
    if len(models) != len(titles):
        raise ValueError("models and titles must have the same length")

    if quantiles is None:
        quantiles = np.linspace(0, 0.8, 6)
    quantiles = np.asarray(quantiles)
    n_q = len(quantiles)
    n_m = len(models)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(n_m)]

    # use test set for these plots
    X = test_dataset.features.to(device)
    y_true = test_dataset.labels.cpu().numpy().ravel()

    # prepare figure
    fig, axes = plt.subplots(nrows=n_q, ncols=n_m, figsize=(4 * n_m, 3 * n_q), squeeze=False)

    for j, (m, title) in enumerate(zip(models, titles)):
        m.to(device)
        m.eval()
        with torch.no_grad():
            pred, uncertainty = m.predict(X)
            # support models that return only prediction
            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]
            pred = pred.cpu().numpy().ravel()
            if uncertainty is not None:
                uncertainty = uncertainty.cpu().numpy().ravel()
        # compute common limits for nicer view
        all_vals = np.concatenate([y_true, pred])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (vmax - vmin) * 0.03 if vmax > vmin else 0.1
        lims = (vmin - pad, vmax + pad)

        color = colors[j]

        for i_q, q in enumerate(quantiles):
            ax = axes[i_q, j]

            if uncertainty is None:
                # if no uncertainty available, mark all as "low" (alpha 0.7)
                mask_low = np.ones_like(y_true, dtype=bool)
            else:
                thresh = np.quantile(uncertainty, 1-q)
                mask_low = uncertainty <= thresh

            # low uncertainty points
            ax.scatter(y_true[mask_low], pred[mask_low], color=color, alpha=0.7, s=markersize)
            # high uncertainty points
            if np.any(~mask_low):
                ax.scatter(y_true[~mask_low], pred[~mask_low], color=color, alpha=0.05, s=markersize)

            # diagonal
            ax.plot(lims, lims, 'k--', linewidth=0.8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # labels: only left column gets y label, only bottom row gets x label
            if j == 0:
                ax.set_ylabel(f"Predicted")
            if i_q == n_q - 1:
                ax.set_xlabel("True")

            # titles on top row
            if i_q == 0:
                ax.set_title(title)

            # annotate quantile on the left of the row
            if j == 0:
                ax.text(0.01, 0.95, f"confidence={q:.2f}", transform=ax.transAxes, va='top', ha='left', fontsize=20, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            ax.grid(True)

    plt.tight_layout()
    plt.show()


