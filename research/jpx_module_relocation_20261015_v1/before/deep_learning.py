import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from time import time
from pathlib import Path

import deep_learning
from data_loader import main_loader
import data_processor
import features
import evaluation


#================================== models ==================================#
class CNN_LSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_out_channels: int=64,
        kernel_size: int=1,
        hidden_size=100
    ):
        self.kernel_size = kernel_size
        
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)
        
        self.dropout1d = nn.Dropout1d(p=0.2) # Make the whole dimension value = 0
        self.dropout = nn.Dropout(p=0.2) # Make single value = 0
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=hidden_size, 
            num_layers=3, 
            batch_first=True, 
            dropout=0.2
        )
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input:    (batch, time, features)
        # required: (batch, features, time)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.relu(x)
        
        # If time_span = 1 --> (batch, features, time) time only equals 1
        # It cannot do pool with kernel 2 cuz only 1 value in that time
        if x.shape[-1] >= self.kernel_size: # time
            x = self.pool(x)
        
        # CNN let some vector channels go to 0 -> decrease dependencies on certain channel vector
        x = self.dropout1d(x) 
        
        # LSTM batch first=True --> (batch, time, features)
        # But now is (batch, channels, time)
        x = x.transpose(1, 2)
        
        output, (h_n, c_n) = self.lstm(x)
        
        # Take the last layer, last time period output
        x = output[:, -1, :] # (batch, 100) (time is removed)
        
        # Can only drop one value, if use dropout1d, only remains batch with all 0 values
        x = self.dropout(x)
        prediction = self.linear(x)
        
        return prediction
        

def make_it_tensor(
    array: list | np.ndarray | pd.Series | pd.DataFrame
) -> torch.tensor:
    if isinstance(array, list):
        array = np.array(array)
    
    if isinstance(array, pd.Series) or isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    
    
    return torch.from_numpy(array.astype(np.float32))
    

def make_loader(X, y, batch_size=64, shuffle=False):
    # DataFrame -> Numpy -> Tensor (DataFrame cannot change to Tensor directly)
    X = make_it_tensor(X)
    y = make_it_tensor(y)
    
    # (samples,) -> (samples, 1)
    # For matching the dimension of output from CNN-LSTM of (64, 1) 
    # While original y's shape is (64,) only
    if y.ndim == 1:
        y = y.unsqueeze(1)
        
    # Match answer with preidction
    dataset = TensorDataset(X, y)
    
    # Make it a dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def train(model, data_loader, loss_fn, optimizer, epoch=1, device="cpu"):
    model.train()
    model = model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        prediction = model(X_batch)
        
        loss = loss_fn(prediction, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
        batch_size = X_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_sample += batch_size
    
    if total_sample == 0:
        raise ValueError("Missing data (sample) for training")
    
    return total_loss / total_sample 


def evaluation(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_fn: nn.Module,
    device: str = "cpu"
) -> tuple[float, np.ndarray]:
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    # For evaluation.py's df
    eval_predictions = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            
            # reshape(-1) is a wild card that flattens multi dimension 
            # inputs into 1 dimension
            eval_predictions.append(predictions.cpu().reshape(-1))
            
            loss = loss_fn(predictions, y_batch)
            
            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_sample += batch_size
    
    # input: [[1.0], [2.0].....] --> concat and change to ndarray()
    if total_sample == 0 or len(eval_predictions) == 0: 
        raise ValueError("Required more data to get the desired results")
    
    all_predictions = torch.cat(eval_predictions).numpy()
    
    return total_loss / total_sample, all_predictions



def print_running_time(func):
    def inner_func():
        print("\n============================== Program starts running ==============================\n")
        start_time = time.time()
        func()
        end_time = time.time()
        print("\n============================== Program finished running ==============================\n")
        diff = end_time - start_time
        hr = diff // 3600
        min = (diff - hr * 3600) // 60
        sec = diff - hr * 3600 - min * 60
        print(f"Total runtime: {hr}h, {min}m, {sec}s")
    return inner_func


def common_process(
    train_and_val: list[str],
    col_names: list[str]
):
    df = main_loader("/Users/coolguy/Desktop/JPX/data/stock_prices.csv", show_basic_data=False)
    
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    print("Stage 1: Loaded", flush=True)

    df = features.get_valid_rows(df, set(col_names))
    
    print("Stage 2: Validated", flush=True)

    df = data_processor.correct_prices_with_factors(df)

    print("Stage 3: Prices adjusted", flush=True)
    
    df, mean, std = features.normalize_data(
        df,
        ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
    )
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Stage 4: Normalized", flush=True)
    
    # Get train date and evaluation date
    train_start, train_end, val_start, val_end = train_and_val
    train_df = data_processor.get_time_for_specific_period(df, train_start, train_end)
    val_df = data_processor.get_time_for_specific_period(df, val_start, val_end)
    
    
    return train_df, val_df, device
        

def save_model(
    model,
    model_name,
    mean,
    std,
    train_start,
    train_end,
    feature_columns,
):
    model_path = Path("checkpoints") / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
            "feature_columns": list(feature_columns),
            "train_start": train_start,
            "train_end": train_end,
        },
        model_path,
    )

    print(f"Model is saved: {model_path.resolve()}")
    return model_path


def load_model(
    initial_model,
    model_path=str,
    map_location="cpu",
    weights_only=True,
    device="cpu",
):
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model path doesn't exist: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=map_location,
        weights_only=weights_only,
    )

    initial_model.load_state_dict(checkpoint["model_state_dict"])
    model = initial_model.to(device)
    model.eval()

    return (
        model,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
        checkpoint["feature_columns"],
        checkpoint["train_start"],
        checkpoint["train_end"],
    )
    