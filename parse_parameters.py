import numpy as np
import pandas as pd
import torch
import os

def parse_key_value_string(s):
    """
    Parse a space-separated string of key=value pairs into a dictionary.
    
    Args:
        s (str): Input string with key=value pairs
    
    Returns:
        dict: Dictionary of parsed key-value pairs
    """
    # Split the string into key-value pairs
    pairs = s.split()
    
    # Create a dictionary of parsed values
    parsed_dict = {}
    for pair in pairs:
        try:
            key, value = pair.split('=')
            # Try to convert to float, keep as string if not possible
            try:
                parsed_dict[key] = float(value)
            except ValueError:
                parsed_dict[key] = value
        except ValueError:
            # Skip any malformed pairs
            continue
    
    return parsed_dict

def read_csvs(data_path):
    # read data
    if os.path.exists(f"{data_path}/val.csv"):
        dfs = []
        ns = []
        for val in ["val", "test", "train"]:
            dfi = pd.read_csv(f"{data_path}/{val}.csv")
            dfs.append(dfi)
            ns.append(dfi.shape[0])
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
    else:
        print("Data folder is not correct or the data was not found.")
        exit()
    return df, ns

def generate_masks(df, params):
    
    # generate masks for different models
    n_classes = len(df.target.unique())
    model_masks = []
    for i in range(0,n_classes):
        # first row for each model
        rowi = params[df.target==i].reset_index(drop=True).iloc[0]
        model_masks.append(list(rowi.isna()))
    model_masks = torch.tensor(model_masks)
    
    return model_masks
    
def prepare_dfs(df):
    # parse string variables
    df_parsed = df['sim_pars'].apply(parse_key_value_string).apply(pd.Series)

    # separate instrument parameters from model parameters
    inst_params = df_parsed[["Lam", "zdepth","InstSetting", "SlitSetting"]]
    for var in ["InstSetting","SlitSetting"]:
        inst_params[var] = inst_params[var].astype(int)

    # one hot encoding of instrument parameters, 10 variables
    inst_params_encoded = pd.get_dummies(inst_params.astype(str))
    inst_encoding = torch.tensor(inst_params_encoded.to_numpy(), dtype=torch.float32)

    # retrieve model parameters
    model_params = df_parsed.drop(["Lam", "zdepth","InstSetting", "SlitSetting"],axis=1)
    model_params_normalized = (model_params - model_params.min(skipna=True)) / (model_params.max(skipna=True) - model_params.min(skipna=True))

    # generate masks
    model_masks = generate_masks(df, model_params_normalized)

    # in principle we can fiil NaNs with 0, since then we mask, and we don't use them
    # Replace NaNs with 0
    model_params_normalized = model_params_normalized.fillna(0)

    
    model_params_normalized =  torch.tensor(model_params_normalized.to_numpy(), dtype=torch.float32)
    
    return inst_encoding, model_params_normalized, model_masks


    
def read_model_parameters(data_path):

    df, ns = read_csvs(data_path)

    inst_encoding, model_params_normalized = prepare_dfs(df)

    model_params_masks = generate_masks(df, model_params_normalized)

    return df, inst_encoding, model_params_normalized, model_params_masks

def masked_mse_loss(y_pred, y, model_mask):
    n_pars = torch.sum(~model_mask)
    return torch.sum(torch.pow(y_pred-y,2) * ~ model_mask) / n_pars
    

        