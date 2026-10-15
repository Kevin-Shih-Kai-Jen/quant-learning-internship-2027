import torch
from torch import nn

import deep_learning
import models
import features
import evaluation

COL_NAMES = ["SecuritiesCode", "Date", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "Target"]
FEAUTRES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
TIME_SPAN = [42]

    
@deep_learning.print_running_time
def model_1():  
    df, _, device = deep_learning.common_process(["2017-01-01", "2021-07-01","2017-01-01", "2021-07-01"], COL_NAMES)
    
    for time in TIME_SPAN:
        X, y = features.create_time_sequence(df, time_sequence=time, col_names=FEAUTRES)

        print("Stage 5: Created time sequence", flush=True)
        data_loader = deep_learning.make_loader(X, y)
        print("Stage 6: Data loaded", flush=True)
        model = models.CNN_LSTM(
            num_features=len(FEAUTRES),
            cnn_out_channels=64,
            kernel_size=time,
            hidden_size=100
        )
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())
        
        train_avg_loss = deep_learning.train(model, data_loader, loss_fn, optimizer, epoch=1, device=device)
        print(f"CNN-LSTM {time} time span training average loss: {train_avg_loss}")
        eval_avg_loss, all_predictions = deep_learning.evaluation(model, data_loader, loss_fn, device)
        print(f"CNN-LSTM {time} time span evaluation average loss: {eval_avg_loss}")

        eval_results = evaluation.overall_performance(df, all_predictions, time)
    
    evaluation.zero_baseline(y)
    
    
def model_2():
    #================================ Expanding window ================================#
    # Half of the year for validation
    splits = [
        ("2017-01-01", "2017-07-01", "2017-07-01", "2018-01-01"), 
        ("2017-01-01", "2018-01-01", "2018-01-01", "2018-07-01"),  
        ("2017-01-01", "2018-07-01", "2018-07-01", "2019-01-01"),  
        ("2017-01-01", "2019-01-01", "2019-01-01", "2019-07-01"),  
        ("2017-01-01", "2019-07-01", "2019-07-01", "2020-01-01"),  
        ("2017-01-01", "2020-01-01", "2020-01-01", "2020-07-01"),  
        ("2017-01-01", "2020-07-01", "2020-07-01", "2021-01-01"),  
        ("2017-01-01", "2021-01-01", "2021-01-01", "2021-07-01"),  
    ]
    
    
    for time in TIME_SPAN:
        # Record all validations results
        total_earnings, track_all_predictions = [], []
        
        model = deep_learning.models(
            num_features=len(FEAUTRES),
            cnn_out_channels=64,
            kernel_size=time,
            hidden_size=100
        )
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for train_and_val in splits:
            train_df, val_df, device = deep_learning.common_process(train_and_val, COL_NAMES)
            
            X_train, y_train = features.create_time_sequence(train_df, time_sequence=time, col_names=FEAUTRES)
            X_val, y_val = features.create_time_sequence(val_df, time_sequence=time, col_names=FEAUTRES)

            print("Stage 5: Created time sequence", flush=True)
            
            train_data_loader = deep_learning.make_loader(X_train, y_train)
            val_data_loader = deep_learning.make_loader(X_val, y_val)

            print("Stage 6: Data loaded", flush=True)
            
            train_avg_loss = deep_learning.train(model, train_data_loader, loss_fn, optimizer, epoch=20, device=device)
            print(f"CNN-LSTM {time} time span training average loss: {train_avg_loss}")
            eval_avg_loss, all_predictions = deep_learning.evaluation(model, val_data_loader, loss_fn, device)
            print(f"CNN-LSTM {time} time span evaluation average loss: {eval_avg_loss}")

            eval_results = evaluation.overall_performance(val_df, all_predictions, time)

            # Will use matplotlib to draw the graph in the future
            total_earnings.append(all_predictions.sum())
            track_all_predictions.append(eval_results)
        
        print(f"Total earnings: {total_earnings}%")
        print(track_all_predictions)
        
        deep_learning.save_model(
            model=model,
            model_name='CNN_LSTM',
            mean=0,
            std=1,
            train_start="2017-01-01",
            train_end="2021-01-01"
        )

            
            
        