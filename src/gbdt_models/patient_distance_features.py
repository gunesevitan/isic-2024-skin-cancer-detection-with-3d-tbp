import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

sys.path.append('..')
import settings


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df['anatom_site_general_filled'] = df['anatom_site_general'].values
    df.loc[df['anatom_site_general_filled'].isnull(), 'anatom_site_general_filled'] = 'Unknown'

    color_columns = [
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_L', 'tbp_lv_Lext',
        'tbp_lv_H', 'tbp_lv_Hext'
    ]
    lesion_color_columns = [
        'tbp_lv_A',
        'tbp_lv_B',
        'tbp_lv_C',
        'tbp_lv_L',
        'tbp_lv_H'
    ]
    outside_color_columns = [
        'tbp_lv_Aext',
        'tbp_lv_Bext',
        'tbp_lv_Cext',
        'tbp_lv_Lext',
        'tbp_lv_Hext'
    ]
    shape_columns = [
        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
    ]
    location_columns = ['tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
    column_groups = [color_columns, lesion_color_columns, outside_color_columns, shape_columns, location_columns]
    column_group_names = ['color', 'lesion_color', 'outside_color', 'shape', 'location']

    difference_and_ratio_columns = [
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext',

        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
        'tbp_lv_area_perim_ratio', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',

        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    ]

    for (patient_id,), df_group in tqdm(df.groupby(['patient_id']), total=1042):

        if df_group.shape[0] > 2:

            idx = df['patient_id'] == patient_id

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                pairwise_distances = squareform(pdist(min_max_scaler.fit_transform(df_group[columns])))
                np.fill_diagonal(pairwise_distances, np.nan)

                mean_distances = np.nanmean(pairwise_distances, axis=1)
                median_distances = np.nanmedian(pairwise_distances, axis=1)
                std_distances = np.nanstd(pairwise_distances, axis=1)
                min_distances = np.nanmin(pairwise_distances, axis=1)
                max_distances = np.nanmax(pairwise_distances, axis=1)
                min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
                max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

                df.loc[idx, f'patient_{column_group_name}_mean_distance'] = mean_distances
                df.loc[idx, f'patient_{column_group_name}_median_distance'] = median_distances
                df.loc[idx, f'patient_{column_group_name}_std_distance'] = std_distances
                df.loc[idx, f'patient_{column_group_name}_min_distance'] = min_distances
                df.loc[idx, f'patient_{column_group_name}_max_distance'] = max_distances
                df.loc[idx, f'patient_{column_group_name}_min_distance_idx'] = min_distance_idx
                df.loc[idx, f'patient_{column_group_name}_max_distance_idx'] = max_distance_idx

                df_patient = df.loc[idx].reset_index(drop=True)
                df_patient_min_aligned = df_patient.iloc[df_patient[f'patient_{column_group_name}_min_distance_idx'].values].reset_index(drop=True)
                df_patient_max_aligned = df_patient.iloc[df_patient[f'patient_{column_group_name}_max_distance_idx'].values].reset_index(drop=True)

                df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_{column}_min_{column_group_name}_distance_lesion_difference' for column in df_patient_min_differences.columns})
                df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

                df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_{column}_min_{column_group_name}_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
                df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

                df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_{column}_max_{column_group_name}_distance_lesion_difference' for column in df_patient_max_differences.columns})
                df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

                df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_{column}_max_{column_group_name}_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
                df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                pairwise_distances = squareform(pdist(min_max_scaler.fit_transform(df_group[columns])))
                np.fill_diagonal(pairwise_distances, np.nan)

                mean_distances = np.nanmean(pairwise_distances, axis=1)
                median_distances = np.nanmedian(pairwise_distances, axis=1)
                std_distances = np.nanstd(pairwise_distances, axis=1)
                min_distances = np.nanmin(pairwise_distances, axis=1)
                max_distances = np.nanmax(pairwise_distances, axis=1)
                min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
                max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

                df.loc[idx, f'patient_site_{column_group_name}_mean_distance'] = mean_distances
                df.loc[idx, f'patient_site_{column_group_name}_median_distance'] = median_distances
                df.loc[idx, f'patient_site_{column_group_name}_std_distance'] = std_distances
                df.loc[idx, f'patient_site_{column_group_name}_min_distance'] = min_distances
                df.loc[idx, f'patient_site_{column_group_name}_max_distance'] = max_distances
                df.loc[idx, f'patient_site_{column_group_name}_min_distance_idx'] = min_distance_idx
                df.loc[idx, f'patient_site_{column_group_name}_max_distance_idx'] = max_distance_idx

                df_patient = df.loc[idx].reset_index(drop=True)
                df_patient_min_aligned = df_patient.iloc[df_patient[f'patient_site_{column_group_name}_min_distance_idx'].values].reset_index(drop=True)
                df_patient_max_aligned = df_patient.iloc[df_patient[f'patient_site_{column_group_name}_max_distance_idx'].values].reset_index(drop=True)

                df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_site_{column}_min_{column_group_name}_distance_lesion_difference' for column in df_patient_min_differences.columns})
                df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

                df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_site_{column}_min_{column_group_name}_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
                df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

                df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_site_{column}_max_{column_group_name}_distance_lesion_difference' for column in df_patient_max_differences.columns})
                df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

                df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_site_{column}_max_{column_group_name}_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
                df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, location), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location']), total=13247):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['tbp_lv_location'] == location)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                pairwise_distances = squareform(pdist(min_max_scaler.fit_transform(df_group[columns])))
                np.fill_diagonal(pairwise_distances, np.nan)

                mean_distances = np.nanmean(pairwise_distances, axis=1)
                median_distances = np.nanmedian(pairwise_distances, axis=1)
                std_distances = np.nanstd(pairwise_distances, axis=1)
                min_distances = np.nanmin(pairwise_distances, axis=1)
                max_distances = np.nanmax(pairwise_distances, axis=1)
                min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
                max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

                df.loc[idx, f'patient_location_{column_group_name}_mean_distance'] = mean_distances
                df.loc[idx, f'patient_location_{column_group_name}_median_distance'] = median_distances
                df.loc[idx, f'patient_location_{column_group_name}_std_distance'] = std_distances
                df.loc[idx, f'patient_location_{column_group_name}_min_distance'] = min_distances
                df.loc[idx, f'patient_location_{column_group_name}_max_distance'] = max_distances
                df.loc[idx, f'patient_location_{column_group_name}_min_distance_idx'] = min_distance_idx
                df.loc[idx, f'patient_location_{column_group_name}_max_distance_idx'] = max_distance_idx

                df_patient = df.loc[idx].reset_index(drop=True)
                df_patient_min_aligned = df_patient.iloc[df_patient[f'patient_location_{column_group_name}_min_distance_idx'].values].reset_index(drop=True)
                df_patient_max_aligned = df_patient.iloc[df_patient[f'patient_location_{column_group_name}_max_distance_idx'].values].reset_index(drop=True)

                df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_location_{column}_min_{column_group_name}_distance_lesion_difference' for column in df_patient_min_differences.columns})
                df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

                df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_location_{column}_min_{column_group_name}_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
                df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

                df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_location_{column}_max_{column_group_name}_distance_lesion_difference' for column in df_patient_max_differences.columns})
                df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

                df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_location_{column}_max_{column_group_name}_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
                df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, location_simple), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location_simple']), total=6967):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['tbp_lv_location_simple'] == location_simple)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                pairwise_distances = squareform(pdist(min_max_scaler.fit_transform(df_group[columns])))
                np.fill_diagonal(pairwise_distances, np.nan)

                mean_distances = np.nanmean(pairwise_distances, axis=1)
                median_distances = np.nanmedian(pairwise_distances, axis=1)
                std_distances = np.nanstd(pairwise_distances, axis=1)
                min_distances = np.nanmin(pairwise_distances, axis=1)
                max_distances = np.nanmax(pairwise_distances, axis=1)
                min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
                max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

                df.loc[idx, f'patient_location_simple_{column_group_name}_mean_distance'] = mean_distances
                df.loc[idx, f'patient_location_simple_{column_group_name}_median_distance'] = median_distances
                df.loc[idx, f'patient_location_simple_{column_group_name}_std_distance'] = std_distances
                df.loc[idx, f'patient_location_simple_{column_group_name}_min_distance'] = min_distances
                df.loc[idx, f'patient_location_simple_{column_group_name}_max_distance'] = max_distances
                df.loc[idx, f'patient_location_simple_{column_group_name}_min_distance_idx'] = min_distance_idx
                df.loc[idx, f'patient_location_simple_{column_group_name}_max_distance_idx'] = max_distance_idx

                df_patient = df.loc[idx].reset_index(drop=True)
                df_patient_min_aligned = df_patient.iloc[df_patient[f'patient_location_simple_{column_group_name}_min_distance_idx'].values].reset_index(drop=True)
                df_patient_max_aligned = df_patient.iloc[df_patient[f'patient_location_simple_{column_group_name}_max_distance_idx'].values].reset_index(drop=True)

                df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_location_simple_{column}_min_{column_group_name}_distance_lesion_difference' for column in df_patient_min_differences.columns})
                df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

                df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
                df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_location_simple_{column}_min_{column_group_name}_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
                df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

                df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_location_simple_{column}_max_{column_group_name}_distance_lesion_difference' for column in df_patient_max_differences.columns})
                df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

                df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
                df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_location_simple_{column}_max_{column_group_name}_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
                df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    patient_distance_features = df.columns.tolist()[56:]
    df.loc[:, patient_distance_features].to_parquet(settings.DATA / 'patient_distance_features.parquet')
