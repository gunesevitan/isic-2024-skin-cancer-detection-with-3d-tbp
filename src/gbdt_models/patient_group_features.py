import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM

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
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm',
        'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max', 'tbp_lv_norm_color',
    ]
    shape_columns = [
        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
        'tbp_lv_area_perim_ratio',
    ]
    location_columns = ['tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
    column_groups = [color_columns, shape_columns, location_columns]
    column_group_names = ['color', 'shape', 'location']

    for (patient_id,), df_group in tqdm(df.groupby(['patient_id']), total=1042):

        idx = df['patient_id'] == patient_id

        for columns, column_group_name in zip(column_groups, column_group_names):

            min_max_scaler = MinMaxScaler()
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
            df.loc[idx, f'patient_{column_group_name}_cluster'] = dbscan.labels_

            standard_scaler = StandardScaler()
            X = standard_scaler.fit_transform(df_group[columns])
            one_class_svm = OneClassSVM()
            one_class_svm.fit(X)
            scores = one_class_svm.score_samples(X)
            df.loc[idx, f'patient_{column_group_name}_anomaly_score'] = pd.Series(scores).rank(pct=True).values

    for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

        idx = (df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site)

        for columns, column_group_name in zip(column_groups, column_group_names):

            min_max_scaler = MinMaxScaler()
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
            df.loc[idx, f'patient_site_{column_group_name}_cluster'] = dbscan.labels_

            standard_scaler = StandardScaler()
            X = standard_scaler.fit_transform(df_group[columns])
            one_class_svm = OneClassSVM()
            one_class_svm.fit(X)
            scores = one_class_svm.score_samples(X)
            df.loc[idx, f'patient_site_{column_group_name}_anomaly_score'] = pd.Series(scores).rank(pct=True).values

    df['patient_color_cluster'] = df['patient_color_cluster'].astype(np.uint16)
    df['patient_shape_cluster'] = df['patient_shape_cluster'].astype(np.uint16)
    df['patient_location_cluster'] = df['patient_location_cluster'].astype(np.uint16)
    df['patient_site_color_cluster'] = df['patient_site_color_cluster'].astype(np.uint16)
    df['patient_site_shape_cluster'] = df['patient_site_shape_cluster'].astype(np.uint16)
    df['patient_site_location_cluster'] = df['patient_site_location_cluster'].astype(np.uint16)

    df['patient_color_cluster_count'] = df.groupby(['patient_id', 'patient_color_cluster'])['patient_color_cluster'].transform('count')
    df['patient_shape_cluster_count'] = df.groupby(['patient_id', 'patient_shape_cluster'])['patient_shape_cluster'].transform('count')
    df['patient_location_cluster_count'] = df.groupby(['patient_id', 'patient_location_cluster'])['patient_location_cluster'].transform('count')
    df['patient_site_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_color_cluster'])['patient_color_cluster'].transform('count')
    df['patient_site_shape_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_shape_cluster'])['patient_shape_cluster'].transform('count')
    df['patient_site_location_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_location_cluster'])['patient_location_cluster'].transform('count')

    patient_group_features = [
        'patient_color_cluster', 'patient_shape_cluster', 'patient_location_cluster',
        'patient_site_color_cluster', 'patient_site_shape_cluster', 'patient_site_location_cluster',

        'patient_color_cluster_count', 'patient_shape_cluster_count', 'patient_location_cluster_count',
        'patient_site_color_cluster_count', 'patient_site_shape_cluster_count', 'patient_site_location_cluster_count',

        'patient_color_anomaly_score', 'patient_shape_anomaly_score', 'patient_location_anomaly_score',
        'patient_site_color_anomaly_score', 'patient_site_shape_anomaly_score', 'patient_site_location_anomaly_score',
    ]
    df.loc[:, patient_group_features].to_parquet(settings.DATA / 'patient_group_features.parquet')
