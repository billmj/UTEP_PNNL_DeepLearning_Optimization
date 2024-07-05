import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

import os
import argparse
from utils import FLconfig

# downloaded csv for train and test set interchanged, test-set has 175341 entries and train-set has 82,332
path_to_train_set = "./data/from_unsw_website/UNSW_NB15_training-set.csv"
path_to_test_set = "./data/from_unsw_website/UNSW_NB15_testing-set.csv"
path_to_client_dataset = "./data/client_dataset"

def partition_dataframe(df, number_of_clients):
    shuffled_df = df.sample(frac=1, random_state=42)  # Shuffle the DataFrame randomly

    # Reset the index of the shuffled DataFrame
    df = shuffled_df.reset_index(drop=True)
    # Calculate the size of each partition
    partition_size = (len(df) // number_of_clients) + 1

    # Partition the dataframe
    for idx, i in enumerate(range(0, len(df), partition_size)):
        partition = df[i:i + partition_size]
        partition.to_csv(os.path.join(path_to_client_dataset, f"client_{idx}.csv"), index=False)

def create_categorical_dict_and_change_values_of_dataframe(df_train_test):
    # create dictionary of binary mapping of each column
    categorical_columns_dict = {i: {} for i in FLconfig.categorical_columns_for_dataset}
    for cat_col in categorical_columns_dict.keys():
        unique_entries_for_col = df_train_test[cat_col].unique()
        categorical_columns_dict[cat_col] = {k: v for k, v in zip(unique_entries_for_col, range(len(unique_entries_for_col)))}

    # for train and test dataframe, map the values of each column in dataframe with dictionary and replace
    for cat_col in FLconfig.categorical_columns_for_dataset:
        df_train_test[cat_col] = df_train_test[cat_col].replace(categorical_columns_dict[cat_col])
        df_train_test[cat_col] = df_train_test[cat_col].infer_objects(copy=False)  # Explicitly handle downcasting

    return df_train_test

def main(number_of_clients):
    # Clear the client dataset directory if it exists
    if os.path.exists(path_to_client_dataset):
        for file in os.listdir(path_to_client_dataset):
            file_path = os.path.join(path_to_client_dataset, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    if not os.path.isdir(path_to_client_dataset):
        os.makedirs(path_to_client_dataset, exist_ok=True)

    df = pd.read_csv(path_to_train_set)
    df_test = pd.read_csv(path_to_test_set)
    df_train_test = pd.concat([df, df_test]).copy()

    df_train_test = create_categorical_dict_and_change_values_of_dataframe(df_train_test)

    # Write the centralized datasets to CSV
    df.to_csv(f"./data/centralized_data.csv", index=False)
    df_test.to_csv(f"./data/centralized_test_data.csv", index=False)

    partition_dataframe(df, number_of_clients)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_of_clients", type=int, required=True, help="Number of clients")
    args = parser.parse_args()

    main(args.number_of_clients)
