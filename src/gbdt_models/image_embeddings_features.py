import sys
from tqdm import tqdm
import pickle
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


sys.path.append('..')
import settings



if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df['anatom_site_general_filled'] = df['anatom_site_general'].values
    df.loc[df['anatom_site_general_filled'].isnull(), 'anatom_site_general_filled'] = 'Unknown'

    image_embeddings = np.load(settings.DATA / 'image_embeddings.npy')

    pca = PCA(n_components=4, random_state=42)
    image_embeddings_pca = pca.fit_transform(image_embeddings)

    with open(settings.MODELS / 'encoders' / 'image_embeddings_pca.pickle', mode='wb') as f:
        pickle.dump(image_embeddings_pca, f)

    image_embeddings_features = pd.DataFrame(image_embeddings_pca, columns=[f'image_embedding_{i}' for i in range(1, image_embeddings_pca.shape[1] + 1)])

    image_embeddings_features['image_embedding_mean'] = np.mean(image_embeddings, axis=1)
    image_embeddings_features['image_embedding_median'] = np.median(image_embeddings, axis=1)
    image_embeddings_features['image_embedding_std'] = np.std(image_embeddings, axis=1)
    image_embeddings_features['image_embedding_min'] = np.min(image_embeddings, axis=1)
    image_embeddings_features['image_embedding_max'] = np.max(image_embeddings, axis=1)

    image_embeddings_features.to_parquet(settings.DATA / 'image_embeddings_features.parquet')

    patient_ids = np.unique(df['patient_id'].values)

    image_embeddings = torch.as_tensor(image_embeddings, device='cuda')

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

            idx = np.where(df['patient_id'] == patient_id)[0]

            patient_image_embeddings = image_embeddings[idx]
            pairwise_distances = (patient_image_embeddings @ patient_image_embeddings.T).cpu().numpy()
            np.fill_diagonal(pairwise_distances, np.nan)

            mean_distances = np.nanmean(pairwise_distances, axis=1)
            median_distances = np.nanmedian(pairwise_distances, axis=1)
            std_distances = np.nanstd(pairwise_distances, axis=1)
            min_distances = np.nanmin(pairwise_distances, axis=1)
            max_distances = np.nanmax(pairwise_distances, axis=1)
            min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
            max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

            df.loc[idx, 'patient_image_mean_distance'] = mean_distances
            df.loc[idx, 'patient_image_median_distance'] = median_distances
            df.loc[idx, 'patient_image_std_distance'] = std_distances
            df.loc[idx, 'patient_image_min_distance'] = min_distances
            df.loc[idx, 'patient_image_max_distance'] = max_distances
            df.loc[idx, 'patient_image_min_distance_idx'] = min_distance_idx
            df.loc[idx, 'patient_image_max_distance_idx'] = max_distance_idx

            df_patient = df.loc[idx].reset_index(drop=True)
            df_patient_min_aligned = df_patient.iloc[df_patient['patient_image_min_distance_idx'].values].reset_index(drop=True)
            df_patient_max_aligned = df_patient.iloc[df_patient['patient_image_max_distance_idx'].values].reset_index(drop=True)

            df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_{column}_min_image_distance_lesion_difference' for column in df_patient_min_differences.columns})
            df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

            df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_{column}_min_image_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
            df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

            df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_{column}_max_image_distance_lesion_difference' for column in df_patient_max_differences.columns})
            df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

            df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_{column}_max_image_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
            df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

        if df_group.shape[0] > 2:

            idx = np.where((df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site))[0]

            patient_image_embeddings = image_embeddings[idx]
            pairwise_distances = (patient_image_embeddings @ patient_image_embeddings.T).cpu().numpy()
            np.fill_diagonal(pairwise_distances, np.nan)

            mean_distances = np.nanmean(pairwise_distances, axis=1)
            median_distances = np.nanmedian(pairwise_distances, axis=1)
            std_distances = np.nanstd(pairwise_distances, axis=1)
            min_distances = np.nanmin(pairwise_distances, axis=1)
            max_distances = np.nanmax(pairwise_distances, axis=1)
            min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
            max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

            df.loc[idx, 'patient_site_image_mean_distance'] = mean_distances
            df.loc[idx, 'patient_site_image_median_distance'] = median_distances
            df.loc[idx, 'patient_site_image_std_distance'] = std_distances
            df.loc[idx, 'patient_site_image_min_distance'] = min_distances
            df.loc[idx, 'patient_site_image_max_distance'] = max_distances
            df.loc[idx, 'patient_site_image_min_distance_idx'] = min_distance_idx
            df.loc[idx, 'patient_site_image_max_distance_idx'] = max_distance_idx

            df_patient = df.loc[idx].reset_index(drop=True)
            df_patient_min_aligned = df_patient.iloc[df_patient['patient_site_image_min_distance_idx'].values].reset_index(drop=True)
            df_patient_max_aligned = df_patient.iloc[df_patient['patient_site_image_max_distance_idx'].values].reset_index(drop=True)

            df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_site_{column}_min_image_distance_lesion_difference' for column in df_patient_min_differences.columns})
            df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

            df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_site_{column}_min_image_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
            df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

            df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_site_{column}_max_image_distance_lesion_difference' for column in df_patient_max_differences.columns})
            df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

            df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_site_{column}_max_image_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
            df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, location), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location']), total=13247):

        if df_group.shape[0] > 2:

            idx = np.where((df['patient_id'] == patient_id) & (df['tbp_lv_location'] == location))[0]

            patient_image_embeddings = image_embeddings[idx]
            pairwise_distances = (patient_image_embeddings @ patient_image_embeddings.T).cpu().numpy()
            np.fill_diagonal(pairwise_distances, np.nan)

            mean_distances = np.nanmean(pairwise_distances, axis=1)
            median_distances = np.nanmedian(pairwise_distances, axis=1)
            std_distances = np.nanstd(pairwise_distances, axis=1)
            min_distances = np.nanmin(pairwise_distances, axis=1)
            max_distances = np.nanmax(pairwise_distances, axis=1)
            min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
            max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

            df.loc[idx, 'patient_location_image_mean_distance'] = mean_distances
            df.loc[idx, 'patient_location_image_median_distance'] = median_distances
            df.loc[idx, 'patient_location_image_std_distance'] = std_distances
            df.loc[idx, 'patient_location_image_min_distance'] = min_distances
            df.loc[idx, 'patient_location_image_max_distance'] = max_distances
            df.loc[idx, 'patient_location_image_min_distance_idx'] = min_distance_idx
            df.loc[idx, 'patient_location_image_max_distance_idx'] = max_distance_idx

            df_patient = df.loc[idx].reset_index(drop=True)
            df_patient_min_aligned = df_patient.iloc[df_patient['patient_location_image_min_distance_idx'].values].reset_index(drop=True)
            df_patient_max_aligned = df_patient.iloc[df_patient['patient_location_image_max_distance_idx'].values].reset_index(drop=True)

            df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_location_{column}_min_image_distance_lesion_difference' for column in df_patient_min_differences.columns})
            df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

            df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_location_{column}_min_image_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
            df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

            df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_location_{column}_max_image_distance_lesion_difference' for column in df_patient_max_differences.columns})
            df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

            df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_location_{column}_max_image_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
            df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    for (patient_id, location_simple), df_group in tqdm(df.groupby(['patient_id', 'tbp_lv_location_simple']), total=6967):

        if df_group.shape[0] > 2:

            idx = np.where((df['patient_id'] == patient_id) & (df['tbp_lv_location_simple'] == location_simple))[0]

            patient_image_embeddings = image_embeddings[idx]
            pairwise_distances = (patient_image_embeddings @ patient_image_embeddings.T).cpu().numpy()
            np.fill_diagonal(pairwise_distances, np.nan)

            mean_distances = np.nanmean(pairwise_distances, axis=1)
            median_distances = np.nanmedian(pairwise_distances, axis=1)
            std_distances = np.nanstd(pairwise_distances, axis=1)
            min_distances = np.nanmin(pairwise_distances, axis=1)
            max_distances = np.nanmax(pairwise_distances, axis=1)
            min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
            max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

            df.loc[idx, 'patient_location_simple_image_mean_distance'] = mean_distances
            df.loc[idx, 'patient_location_simple_image_median_distance'] = median_distances
            df.loc[idx, 'patient_location_simple_image_std_distance'] = std_distances
            df.loc[idx, 'patient_location_simple_image_min_distance'] = min_distances
            df.loc[idx, 'patient_location_simple_image_max_distance'] = max_distances
            df.loc[idx, 'patient_location_simple_image_min_distance_idx'] = min_distance_idx
            df.loc[idx, 'patient_location_simple_image_max_distance_idx'] = max_distance_idx

            df_patient = df.loc[idx].reset_index(drop=True)
            df_patient_min_aligned = df_patient.iloc[df_patient['patient_location_simple_image_min_distance_idx'].values].reset_index(drop=True)
            df_patient_max_aligned = df_patient.iloc[df_patient['patient_location_simple_image_max_distance_idx'].values].reset_index(drop=True)

            df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_location_simple_{column}_min_image_distance_lesion_difference' for column in df_patient_min_differences.columns})
            df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

            df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_location_simple_{column}_min_image_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
            df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

            df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_location_simple_{column}_max_image_distance_lesion_difference' for column in df_patient_max_differences.columns})
            df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

            df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_location_simple_{column}_max_image_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
            df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values

    image_distance_features = df.columns.tolist()[56:]
    df.loc[:, image_distance_features].to_parquet(settings.DATA / 'image_distance_features.parquet')
