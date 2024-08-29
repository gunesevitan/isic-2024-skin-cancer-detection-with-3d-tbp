import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

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

    for (patient_id,), df_group in tqdm(df.groupby(['patient_id']), total=1042):

        if df_group.shape[0] > 2:

            idx = df['patient_id'] == patient_id

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
                df.loc[idx, f'patient_{column_group_name}_cluster'] = dbscan.labels_

    for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                dbscan = DBSCAN(eps=0.4, min_samples=1)
                dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
                df.loc[idx, f'patient_site_{column_group_name}_cluster'] = dbscan.labels_

    for (patient_id, location), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location']), total=13247):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['tbp_lv_location'] == location)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
                df.loc[idx, f'patient_location_{column_group_name}_cluster'] = dbscan.labels_

    for (patient_id, location_simple), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location_simple']), total=6967):

        if df_group.shape[0] > 2:

            idx = (df['patient_id'] == patient_id) & (df['tbp_lv_location_simple'] == location_simple)

            for columns, column_group_name in zip(column_groups, column_group_names):

                min_max_scaler = MinMaxScaler()
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
                df.loc[idx, f'patient_location_simple_{column_group_name}_cluster'] = dbscan.labels_

    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')
    df['patient_site_lesion_count'] = df.groupby(['patient_id', 'anatom_site_general_filled'])['patient_id'].transform('count')
    df['patient_location_lesion_count'] = df.groupby(['patient_id', 'tbp_lv_location'])['patient_id'].transform('count')
    df['patient_location_simple_lesion_count'] = df.groupby(['patient_id', 'tbp_lv_location_simple'])['patient_id'].transform('count')

    df['patient_color_cluster'] = df['patient_color_cluster'].astype(np.float32)
    df['patient_lesion_color_cluster'] = df['patient_lesion_color_cluster'].astype(np.float32)
    df['patient_outside_color_cluster'] = df['patient_outside_color_cluster'].astype(np.float32)
    df['patient_shape_cluster'] = df['patient_shape_cluster'].astype(np.float32)
    df['patient_location_cluster'] = df['patient_location_cluster'].astype(np.float32)
    df['patient_site_color_cluster'] = df['patient_site_color_cluster'].astype(np.float32)
    df['patient_site_lesion_color_cluster'] = df['patient_site_lesion_color_cluster'].astype(np.float32)
    df['patient_site_outside_color_cluster'] = df['patient_site_outside_color_cluster'].astype(np.float32)
    df['patient_site_shape_cluster'] = df['patient_site_shape_cluster'].astype(np.float32)
    df['patient_site_location_cluster'] = df['patient_site_location_cluster'].astype(np.float32)
    df['patient_location_color_cluster'] = df['patient_location_color_cluster'].astype(np.float32)
    df['patient_location_lesion_color_cluster'] = df['patient_location_lesion_color_cluster'].astype(np.float32)
    df['patient_location_outside_color_cluster'] = df['patient_location_outside_color_cluster'].astype(np.float32)
    df['patient_location_shape_cluster'] = df['patient_location_shape_cluster'].astype(np.float32)
    df['patient_location_location_cluster'] = df['patient_location_location_cluster'].astype(np.float32)
    df['patient_location_simple_color_cluster'] = df['patient_location_simple_color_cluster'].astype(np.float32)
    df['patient_location_simple_lesion_color_cluster'] = df['patient_location_simple_lesion_color_cluster'].astype(np.float32)
    df['patient_location_simple_outside_color_cluster'] = df['patient_location_simple_outside_color_cluster'].astype(np.float32)
    df['patient_location_simple_shape_cluster'] = df['patient_location_simple_shape_cluster'].astype(np.float32)
    df['patient_location_simple_location_cluster'] = df['patient_location_simple_location_cluster'].astype(np.float32)

    df['patient_color_cluster_count'] = df.groupby(['patient_id', 'patient_color_cluster'])['patient_color_cluster'].transform('count')
    df['patient_lesion_color_cluster_count'] = df.groupby(['patient_id', 'patient_lesion_color_cluster'])['patient_lesion_color_cluster'].transform('count')
    df['patient_outside_color_cluster_count'] = df.groupby(['patient_id', 'patient_outside_color_cluster'])['patient_outside_color_cluster'].transform('count')
    df['patient_shape_cluster_count'] = df.groupby(['patient_id', 'patient_shape_cluster'])['patient_shape_cluster'].transform('count')
    df['patient_location_cluster_count'] = df.groupby(['patient_id', 'patient_location_cluster'])['patient_location_cluster'].transform('count')
    df['patient_site_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_color_cluster'])['patient_site_color_cluster'].transform('count')
    df['patient_site_lesion_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_lesion_color_cluster'])['patient_site_lesion_color_cluster'].transform('count')
    df['patient_site_outside_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_outside_color_cluster'])['patient_site_outside_color_cluster'].transform('count')
    df['patient_site_shape_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_shape_cluster'])['patient_site_shape_cluster'].transform('count')
    df['patient_site_location_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_location_cluster'])['patient_site_location_cluster'].transform('count')
    df['patient_location_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_color_cluster'])['patient_location_color_cluster'].transform('count')
    df['patient_location_lesion_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_lesion_color_cluster'])['patient_location_lesion_color_cluster'].transform('count')
    df['patient_location_outside_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_outside_color_cluster'])['patient_location_outside_color_cluster'].transform('count')
    df['patient_location_shape_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_shape_cluster'])['patient_location_shape_cluster'].transform('count')
    df['patient_location_location_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_location_cluster'])['patient_location_location_cluster'].transform('count')
    df['patient_location_simple_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_simple_color_cluster'])['patient_location_simple_color_cluster'].transform('count')
    df['patient_location_simple_lesion_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_simple_lesion_color_cluster'])['patient_location_simple_lesion_color_cluster'].transform('count')
    df['patient_location_simple_outside_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_simple_outside_color_cluster'])['patient_location_simple_outside_color_cluster'].transform('count')
    df['patient_location_simple_shape_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_simple_shape_cluster'])['patient_location_simple_shape_cluster'].transform('count')
    df['patient_location_simple_location_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_simple_location_cluster'])['patient_location_simple_location_cluster'].transform('count')

    df['patient_color_cluster_ratio'] = df['patient_color_cluster_count'] / df['patient_lesion_count']
    df['patient_lesion_color_cluster_ratio'] = df['patient_lesion_color_cluster_count'] / df['patient_lesion_count']
    df['patient_outside_color_cluster_ratio'] = df['patient_outside_color_cluster_count'] / df['patient_lesion_count']
    df['patient_shape_cluster_ratio'] = df['patient_shape_cluster_count'] / df['patient_lesion_count']
    df['patient_location_cluster_ratio'] = df['patient_location_cluster_count'] / df['patient_lesion_count']
    df['patient_site_color_cluster_ratio'] = df['patient_site_color_cluster_count'] / df['patient_site_lesion_count']
    df['patient_site_lesion_color_cluster_ratio'] = df['patient_site_lesion_color_cluster_count'] / df['patient_site_lesion_count']
    df['patient_site_outside_color_cluster_ratio'] = df['patient_site_outside_color_cluster_count'] / df['patient_site_lesion_count']
    df['patient_site_shape_cluster_ratio'] = df['patient_site_shape_cluster_count'] / df['patient_site_lesion_count']
    df['patient_site_location_cluster_ratio'] = df['patient_site_location_cluster_count'] / df['patient_site_lesion_count']
    df['patient_location_color_cluster_ratio'] = df['patient_location_color_cluster_count'] / df['patient_location_lesion_count']
    df['patient_location_lesion_color_cluster_ratio'] = df['patient_location_lesion_color_cluster_count'] / df['patient_location_lesion_count']
    df['patient_location_outside_color_cluster_ratio'] = df['patient_location_outside_color_cluster_count'] / df['patient_location_lesion_count']
    df['patient_location_shape_cluster_ratio'] = df['patient_location_shape_cluster_count'] / df['patient_location_lesion_count']
    df['patient_location_location_cluster_ratio'] = df['patient_location_location_cluster_count'] / df['patient_location_lesion_count']
    df['patient_location_simple_color_cluster_ratio'] = df['patient_location_simple_color_cluster_count'] / df['patient_location_simple_lesion_count']
    df['patient_location_simple_lesion_color_cluster_ratio'] = df['patient_location_simple_lesion_color_cluster_count'] / df['patient_location_simple_lesion_count']
    df['patient_location_simple_outside_color_cluster_ratio'] = df['patient_location_simple_outside_color_cluster_count'] / df['patient_location_simple_lesion_count']
    df['patient_location_simple_shape_cluster_ratio'] = df['patient_location_simple_shape_cluster_count'] / df['patient_location_simple_lesion_count']
    df['patient_location_simple_location_cluster_ratio'] = df['patient_location_simple_location_cluster_count'] / df['patient_location_simple_lesion_count']

    df['patient_color_cluster_difference'] = df['patient_color_cluster_count'] - df['patient_lesion_count']
    df['patient_lesion_color_cluster_difference'] = df['patient_lesion_color_cluster_count'] - df['patient_lesion_count']
    df['patient_outside_color_cluster_difference'] = df['patient_outside_color_cluster_count'] - df['patient_lesion_count']
    df['patient_shape_cluster_difference'] = df['patient_shape_cluster_count'] - df['patient_lesion_count']
    df['patient_location_cluster_difference'] = df['patient_location_cluster_count'] - df['patient_lesion_count']
    df['patient_site_color_cluster_difference'] = df['patient_site_color_cluster_count'] - df['patient_site_lesion_count']
    df['patient_site_lesion_color_cluster_difference'] = df['patient_site_lesion_color_cluster_count'] - df['patient_site_lesion_count']
    df['patient_site_outside_color_cluster_difference'] = df['patient_site_outside_color_cluster_count'] - df['patient_site_lesion_count']
    df['patient_site_shape_cluster_difference'] = df['patient_site_shape_cluster_count'] - df['patient_site_lesion_count']
    df['patient_site_location_cluster_difference'] = df['patient_site_location_cluster_count'] - df['patient_site_lesion_count']
    df['patient_location_color_cluster_difference'] = df['patient_location_color_cluster_count'] - df['patient_location_lesion_count']
    df['patient_location_lesion_color_cluster_difference'] = df['patient_location_lesion_color_cluster_count'] - df['patient_location_lesion_count']
    df['patient_location_outside_color_cluster_difference'] = df['patient_location_outside_color_cluster_count'] - df['patient_location_lesion_count']
    df['patient_location_shape_cluster_difference'] = df['patient_location_shape_cluster_count'] - df['patient_location_lesion_count']
    df['patient_location_location_cluster_difference'] = df['patient_location_location_cluster_count'] - df['patient_location_lesion_count']
    df['patient_location_simple_color_cluster_difference'] = df['patient_location_simple_color_cluster_count'] - df['patient_location_simple_lesion_count']
    df['patient_location_simple_lesion_color_cluster_difference'] = df['patient_location_simple_lesion_color_cluster_count'] - df['patient_location_simple_lesion_count']
    df['patient_location_simple_outside_color_cluster_difference'] = df['patient_location_simple_outside_color_cluster_count'] - df['patient_location_simple_lesion_count']
    df['patient_location_simple_shape_cluster_difference'] = df['patient_location_simple_shape_cluster_count'] - df['patient_location_simple_lesion_count']
    df['patient_location_simple_location_cluster_difference'] = df['patient_location_simple_location_cluster_count'] - df['patient_location_simple_lesion_count']

    patient_group_features = df.columns.tolist()[56:]
    df.loc[:, patient_group_features].to_parquet(settings.DATA / 'patient_group_features.parquet')
