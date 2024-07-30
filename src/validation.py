import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import settings


def create_folds(df, stratify_column, group_column, n_splits, shuffle=True, random_state=42, verbose=True):

    """
    Create columns of folds on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given stratify columns

    stratify_column: str
        Name of column to be stratified

    group_column: str
        Name of column to be gruoped

    n_splits: int
        Number of folds (2 <= n_splits)

    shuffle: bool
        Whether to shuffle before split or not

    random_state: int
        Random seed for reproducible results

    verbose: bool
        Verbosity flag

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with created fold columns
    """

    if group_column is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (training_idx, validation_idx) in enumerate(sgkf.split(X=df, y=df[stratify_column], groups=df[group_column]), 1):
            df.loc[training_idx, f'fold{fold}'] = 0
            df.loc[validation_idx, f'fold{fold}'] = 1
            df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (training_idx, validation_idx) in enumerate(skf.split(X=df, y=df[stratify_column]), 1):
            df.loc[training_idx, f'fold{fold}'] = 0
            df.loc[validation_idx, f'fold{fold}'] = 1
            df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    if verbose:

        settings.logger.info(f'Dataset split into {n_splits} folds')

        target_positive_counts = []
        validation_sizes = []

        for fold in range(1, n_splits + 1):
            df_fold = df[df[f'fold{fold}'] == 1]
            stratify_column_value_counts = df_fold[stratify_column].value_counts().to_dict()
            target_value_counts = df_fold['target'].value_counts().to_dict()
            target_positive_counts.append(target_value_counts[1])
            validation_sizes.append(df_fold.shape[0])
            settings.logger.info(f'Fold {fold} {df_fold.shape} - {json.dumps(stratify_column_value_counts, indent=2)} - {json.dumps(target_value_counts, indent=2)}')

        target_positive_counts_std = np.std(target_positive_counts)
        validation_sizes_std = np.std(validation_sizes)
        settings.logger.info(f'seed {random_state} - target {target_positive_counts} {target_positive_counts_std:.4f} - validation {validation_sizes} {validation_sizes_std:.4f}')

    return df


if __name__ == '__main__':

    df_train_metadata = pd.read_parquet(settings.DATA / 'datasets' / 'isic-2024-metadata.parquet')
    settings.logger.info(f'Dataset Shape: {df_train_metadata.shape}')

    df_train_metadata['stratify_column'] = df_train_metadata['iddx_1'].values
    df_train_metadata.loc[df_train_metadata['target'] == 1, 'stratify_column'] = df_train_metadata.loc[df_train_metadata['target'] == 1, 'iddx_3'].values

    n_splits = 5
    df_train_metadata = create_folds(
        df=df_train_metadata,
        stratify_column='stratify_column',
        group_column='patient_id',
        n_splits=n_splits,
        shuffle=True,
        random_state=87,
        verbose=True
    )

    df_isic_master_dataset_metadata = pd.read_parquet(settings.DATA / 'isic_master_dataset' / 'metadata.parquet')
    df_isic_master_dataset_metadata = df_isic_master_dataset_metadata.loc[
        (df_isic_master_dataset_metadata['dataset'] != 'isic_2024') &
        (df_isic_master_dataset_metadata['target'].notna())
    ].reset_index(drop=True)
    df_isic_master_dataset_metadata['stratify_column'] = df_isic_master_dataset_metadata['target'].astype(str) + df_isic_master_dataset_metadata['dataset']
    df_isic_master_dataset_metadata = create_folds(
        df=df_isic_master_dataset_metadata,
        stratify_column='stratify_column',
        group_column=None,
        n_splits=n_splits,
        shuffle=True,
        random_state=0,
        verbose=True
    )

    df_folds = pd.concat((
        df_train_metadata[['isic_id'] + [f'fold{fold}' for fold in range(1, n_splits + 1)]],
        df_isic_master_dataset_metadata[['isic_id'] + [f'fold{fold}' for fold in range(1, n_splits + 1)]]
    ))
    df_folds.to_csv(settings.DATA / 'folds.csv', index=False)
    settings.logger.info(f'folds.csv is saved to {settings.DATA}')
