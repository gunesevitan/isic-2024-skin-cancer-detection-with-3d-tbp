import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import feature_utilities


def create_multiclass_target(df):

    df['target_multiclass'] = df['iddx_1'].map({'Benign': 0, 'Malignant': 1, 'Indeterminate': 2})

    return df


def encode_categorical_columns(df, categorical_columns, encoder_directory):

    """
    Encode given categorical columns

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with LAB color features

    categorical_columns: list
        Array of categorical column names

    encoder_directory: pathlib.Path
        Directory for saving the encoders

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with encoded categorical columns
    """

    for column in categorical_columns:
        encoder = OrdinalEncoder(
            categories='auto',
            dtype=np.float32,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        df[column] = encoder.fit_transform(df[column].values.reshape(-1, 1))

        with open(encoder_directory / f'encoder_{column}.pickle', mode='wb') as f:
            pickle.dump(encoder, f)

    return df


def create_color_features(df):

    """
    Create color features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with LAB color features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with additional color features
    """

    # LAB to RGB and HSV color spaces for lesion and outside
    df['lesion_x'], df['lesion_y'], df['lesion_z'] = feature_utilities.lab_to_xyz(
        l=df['tbp_lv_L'].values,
        a=df['tbp_lv_A'].values,
        b=df['tbp_lv_B'].values
    )
    df['lesion_r'], df['lesion_g'], df['lesion_b'] = feature_utilities.xyz_to_rgb(
        x=df['lesion_x'].values,
        y=df['lesion_y'].values,
        z=df['lesion_z'].values
    )
    df['lesion_h'], df['lesion_s'], df['lesion_v'] = feature_utilities.rgb_to_hsv(
        r=df['lesion_r'].values,
        g=df['lesion_g'].values,
        b=df['lesion_b'].values,
    )
    df['outside_x'], df['outside_y'], df['outside_z'] = feature_utilities.lab_to_xyz(
        l=df['tbp_lv_Lext'].values,
        a=df['tbp_lv_Aext'].values,
        b=df['tbp_lv_Bext'].values
    )
    df['outside_r'], df['outside_g'], df['outside_b'] = feature_utilities.xyz_to_rgb(
        x=df['outside_x'].values,
        y=df['outside_y'].values,
        z=df['outside_z'].values
    )
    df['outside_h'], df['outside_s'], df['outside_v'] = feature_utilities.rgb_to_hsv(
        r=df['outside_r'].values,
        g=df['outside_g'].values,
        b=df['outside_b'].values,
    )

    # LAB space variables weighted sums generated with linear model
    df['lesion_abl_weighted_sum'] = (df['tbp_lv_A'] * 2.8391) + (df['tbp_lv_B'] * -2.7377) + (df['tbp_lv_L'] * 0.2561)
    df['lesion_ch_weighted_sum'] = (df['tbp_lv_C'] * -0.2251) + (df['tbp_lv_H'] * -2.5157)
    df['lesion_abchl_weighted_sum'] = (df['tbp_lv_A'] * -25.6389) + (df['tbp_lv_B'] * -26.4953) + (df['tbp_lv_C'] * 35.7917) + \
                                      (df['tbp_lv_H'] * -5.2130) + (df['tbp_lv_L'] * -0.0733)
    df['outside_abl_weighted_sum'] = (df['tbp_lv_Aext'] * 2.9145) + (df['tbp_lv_Bext'] * -2.090) + (df['tbp_lv_Lext'] * 0.2566)
    df['outside_ch_weighted_sum'] = (df['tbp_lv_Cext'] * -1.8150) + (df['tbp_lv_Hext'] * -0.5339)
    df['outside_abchl_weighted_sum'] = (df['tbp_lv_Aext'] * -26.7866) + (df['tbp_lv_Bext'] * -54.2720) + (df['tbp_lv_Cext'] * 59.8655) + \
                                       (df['tbp_lv_Hext'] * -0.0218) + (df['tbp_lv_Lext'] * 0.1771)
    df['lesion_outside_abchl_weighted_sum'] = (df['tbp_lv_A'] * -26.4163) + (df['tbp_lv_B'] * -20.8240) + (df['tbp_lv_C'] * 29.1262) + \
                                              (df['tbp_lv_H'] * -6.6721) + (df['tbp_lv_L'] * -1.1962) +\
                                              (df['tbp_lv_Aext'] * -6.3414) + (df['tbp_lv_Bext'] * -36.1009) + (df['tbp_lv_Cext'] * 37.4128) + \
                                              (df['tbp_lv_Hext'] * 5.6834) + (df['tbp_lv_Lext'] * 1.3754)

    # LAB space variable interactions for lesion
    df['lesion_ab_ratio'] = df['tbp_lv_A'] / df['tbp_lv_B']
    df['lesion_ab_difference'] = df['tbp_lv_A'] - df['tbp_lv_B']
    df['lesion_ac_ratio'] = df['tbp_lv_A'] / df['tbp_lv_C']
    df['lesion_ac_difference'] = df['tbp_lv_A'] - df['tbp_lv_C']
    df['lesion_ah_ratio'] = df['tbp_lv_A'] / df['tbp_lv_H']
    df['lesion_ah_difference'] = df['tbp_lv_A'] - df['tbp_lv_H']
    df['lesion_al_ratio'] = df['tbp_lv_A'] / df['tbp_lv_L']
    df['lesion_al_difference'] = df['tbp_lv_A'] - df['tbp_lv_L']
    df['lesion_bc_ratio'] = df['tbp_lv_B'] / df['tbp_lv_C']
    df['lesion_bc_difference'] = df['tbp_lv_B'] - df['tbp_lv_C']
    df['lesion_bh_ratio'] = df['tbp_lv_B'] / df['tbp_lv_H']
    df['lesion_bh_difference'] = df['tbp_lv_B'] - df['tbp_lv_H']
    df['lesion_bl_ratio'] = df['tbp_lv_B'] / df['tbp_lv_L']
    df['lesion_bl_difference'] = df['tbp_lv_B'] - df['tbp_lv_L']
    df['lesion_ch_ratio'] = df['tbp_lv_C'] / df['tbp_lv_H']
    df['lesion_ch_difference'] = df['tbp_lv_C'] - df['tbp_lv_H']
    df['lesion_cl_ratio'] = df['tbp_lv_C'] / df['tbp_lv_L']
    df['lesion_cl_difference'] = df['tbp_lv_C'] - df['tbp_lv_L']
    df['lesion_hl_ratio'] = df['tbp_lv_H'] / df['tbp_lv_L']
    df['lesion_hl_difference'] = df['tbp_lv_H'] - df['tbp_lv_L']

    # LAB space variable interactions for outside
    df['outside_ab_ratio'] = df['tbp_lv_Aext'] / df['tbp_lv_Bext']
    df['outside_ab_difference'] = df['tbp_lv_Aext'] - df['tbp_lv_Bext']
    df['outside_ac_ratio'] = df['tbp_lv_Aext'] / df['tbp_lv_Cext']
    df['outside_ac_difference'] = df['tbp_lv_Aext'] - df['tbp_lv_Cext']
    df['outside_ah_ratio'] = df['tbp_lv_Aext'] / df['tbp_lv_Hext']
    df['outside_ah_difference'] = df['tbp_lv_Aext'] - df['tbp_lv_Hext']
    df['outside_al_ratio'] = df['tbp_lv_Aext'] / df['tbp_lv_Lext']
    df['outside_al_difference'] = df['tbp_lv_Aext'] - df['tbp_lv_Lext']
    df['outside_bc_ratio'] = df['tbp_lv_Bext'] / df['tbp_lv_Cext']
    df['outside_bc_difference'] = df['tbp_lv_Bext'] - df['tbp_lv_Cext']
    df['outside_bh_ratio'] = df['tbp_lv_Bext'] / df['tbp_lv_Hext']
    df['outside_bh_difference'] = df['tbp_lv_Bext'] - df['tbp_lv_Hext']
    df['outside_bl_ratio'] = df['tbp_lv_Bext'] / df['tbp_lv_Lext']
    df['outside_bl_difference'] = df['tbp_lv_Bext'] - df['tbp_lv_Lext']
    df['outside_ch_ratio'] = df['tbp_lv_Cext'] / df['tbp_lv_Hext']
    df['outside_ch_difference'] = df['tbp_lv_Cext'] - df['tbp_lv_Hext']
    df['outside_cl_ratio'] = df['tbp_lv_Cext'] / df['tbp_lv_Lext']
    df['outside_cl_difference'] = df['tbp_lv_Cext'] - df['tbp_lv_Lext']
    df['outside_hl_ratio'] = df['tbp_lv_Hext'] / df['tbp_lv_Lext']
    df['outside_hl_difference'] = df['tbp_lv_Hext'] - df['tbp_lv_Lext']

    # LAB space variable interactions interactions for lesion and outside
    df['lesion_outside_ab_difference_ratio'] = df['lesion_ab_difference'] / df['outside_ab_difference']
    df['lesion_outside_ac_difference_ratio'] = df['lesion_ac_difference'] / df['outside_ac_difference']
    df['lesion_outside_ah_difference_ratio'] = df['lesion_ah_difference'] / df['outside_ah_difference']
    df['lesion_outside_al_difference_ratio'] = df['lesion_al_difference'] / df['outside_al_difference']
    df['lesion_outside_bc_difference_ratio'] = df['lesion_bc_difference'] / df['outside_bc_difference']
    df['lesion_outside_bh_difference_ratio'] = df['lesion_bh_difference'] / df['outside_bh_difference']
    df['lesion_outside_bl_difference_ratio'] = df['lesion_bl_difference'] / df['outside_bl_difference']
    df['lesion_outside_ch_difference_ratio'] = df['lesion_ch_difference'] / df['outside_ch_difference']
    df['lesion_outside_cl_difference_ratio'] = df['lesion_cl_difference'] / df['outside_cl_difference']
    df['lesion_outside_hl_difference_ratio'] = df['lesion_hl_difference'] / df['outside_hl_difference']

    # LAB space variable interactions for lesion and outside
    df['a_lesion_outside_ratio'] = df['tbp_lv_A'] / df['tbp_lv_Aext']
    df['a_lesion_outside_difference'] = df['tbp_lv_A'] - df['tbp_lv_Aext']
    df['b_lesion_outside_ratio'] = df['tbp_lv_B'] / df['tbp_lv_Bext']
    df['b_lesion_outside_difference'] = df['tbp_lv_B'] - df['tbp_lv_Bext']
    df['c_lesion_outside_ratio'] = df['tbp_lv_C'] / df['tbp_lv_Cext']
    df['c_lesion_outside_difference'] = df['tbp_lv_C'] - df['tbp_lv_Cext']
    df['h_lesion_outside_ratio'] = df['tbp_lv_H'] / df['tbp_lv_Hext']
    df['h_lesion_outside_difference'] = df['tbp_lv_H'] - df['tbp_lv_Hext']
    df['l_lesion_outside_ratio'] = df['tbp_lv_L'] / df['tbp_lv_Lext']
    df['l_lesion_outside_difference'] = df['tbp_lv_L'] - df['tbp_lv_Lext']
    df['stdl_lesion_outside_ratio'] = df['tbp_lv_stdL'] / df['tbp_lv_stdLExt']
    df['stdl_lesion_outside_difference'] = df['tbp_lv_stdL'] - df['tbp_lv_stdLExt']

    # LAB space dela variable interactions for lesion and surrounding skin
    df['delta_albnorm_ratio'] = df['tbp_lv_deltaA'] / df['tbp_lv_deltaLBnorm']
    df['delta_albnorm_difference'] = df['tbp_lv_deltaA'] - df['tbp_lv_deltaLBnorm']
    df['delta_blbnorm_ratio'] = df['tbp_lv_deltaB'] / df['tbp_lv_deltaLBnorm']
    df['delta_blbnorm_difference'] = df['tbp_lv_deltaB'] - df['tbp_lv_deltaLBnorm']
    df['delta_llbnorm_ratio'] = df['tbp_lv_deltaL'] / df['tbp_lv_deltaLBnorm']
    df['delta_llbnorm_difference'] = df['tbp_lv_deltaL'] - df['tbp_lv_deltaLBnorm']

    return df


def create_shape_features(df):

    """
    Create shape features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with shape features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with additional shape features
    """

    # Diameter interactions
    df['lesion_diameter_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_minorAxisMM']
    df['lesion_diameter_difference'] = df['clin_size_long_diam_mm'] - df['tbp_lv_minorAxisMM']
    df['lesion_diameter_mean'] = (df['clin_size_long_diam_mm'] + df['tbp_lv_minorAxisMM']) / 2

    # Diameter x perimeter interactions
    df['lesion_max_diameter_perimeter_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_perimeterMM']
    df['lesion_max_diameter_perimeter_difference'] = df['clin_size_long_diam_mm'] - df['tbp_lv_perimeterMM']
    df['lesion_min_diameter_perimeter_ratio'] = df['tbp_lv_minorAxisMM'] / df['tbp_lv_perimeterMM']
    df['lesion_min_diameter_perimeter_difference'] = df['tbp_lv_minorAxisMM'] - df['tbp_lv_perimeterMM']
    df['lesion_mean_diameter_perimeter_ratio'] = df['lesion_diameter_mean'] / df['tbp_lv_perimeterMM']
    df['lesion_mean_diameter_perimeter_difference'] = df['lesion_diameter_mean'] - df['tbp_lv_perimeterMM']

    # Area x diameter interactions
    df['lesion_area_max_diameter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['clin_size_long_diam_mm'] ** 2)
    df['lesion_area_max_diameter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['clin_size_long_diam_mm'] ** 2)
    df['lesion_area_min_diameter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_minorAxisMM'] ** 2)
    df['lesion_area_min_diameter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['tbp_lv_minorAxisMM'] ** 2)
    df['lesion_area_mean_diameter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['lesion_diameter_mean'] ** 2)
    df['lesion_area_mean_diameter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['lesion_diameter_mean'] ** 2)

    # Area x perimeter interaction
    df['lesion_area_perimeter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2)
    df['lesion_area_perimeter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['tbp_lv_perimeterMM'] ** 2)
    df['lesion_circularity'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM'] ** 2)
    df['lesion_shape_index'] = df['tbp_lv_perimeterMM'] / np.sqrt(df['tbp_lv_areaMM2'])

    return df


def create_coordinate_features(df):

    """
    Create coordinate features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with coordinate features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with additional coordinate features
    """

    df['lesion_distance_to_origin'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
    df['lesion_angle_xy'] = np.arctan2(df['tbp_lv_x'], df['tbp_lv_y'])
    df['lesion_angle_xz'] = np.arctan2(df['tbp_lv_x'], df['tbp_lv_z'])
    df['lesion_angle_yz'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_z'])

    return df


def create_irregularity_features(df):

    """
    Create irregularity features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with raw features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with additional irregularity features
    """

    border_irregularity_columns = ['tbp_lv_norm_border', 'tbp_lv_symm_2axis', 'tbp_lv_area_perim_ratio']
    df['border_irregularity_mean'] = df[border_irregularity_columns].mean(axis=1)
    df['border_irregularity_std'] = df[border_irregularity_columns].std(axis=1)
    df['border_irregularity_min'] = df[border_irregularity_columns].min(axis=1)
    df['border_irregularity_max'] = df[border_irregularity_columns].max(axis=1)
    df['border_irregularity_sum'] = df[border_irregularity_columns].sum(axis=1)

    color_irregularity_columns = ['tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max']
    df['color_irregularity_mean'] = df[color_irregularity_columns].mean(axis=1)
    df['color_irregularity_std'] = df[color_irregularity_columns].std(axis=1)
    df['color_irregularity_min'] = df[color_irregularity_columns].min(axis=1)
    df['color_irregularity_max'] = df[color_irregularity_columns].max(axis=1)
    df['color_irregularity_sum'] = df[color_irregularity_columns].sum(axis=1)

    irregularity_columns = border_irregularity_columns + color_irregularity_columns
    df['irregularity_mean'] = df[irregularity_columns].mean(axis=1)
    df['irregularity_std'] = df[irregularity_columns].std(axis=1)
    df['irregularity_min'] = df[irregularity_columns].min(axis=1)
    df['irregularity_max'] = df[irregularity_columns].max(axis=1)
    df['irregularity_sum'] = df[irregularity_columns].sum(axis=1)
    df['irregularity_skew'] = df[irregularity_columns].skew(axis=1)

    return df


def create_aggregation_features(df):

    """
    Create aggregation features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with raw features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with aggregation features
    """

    df['patient_normalized_isic_id'] = df['isic_id'].apply(lambda x: int(x.split('_')[-1]))
    df['patient_normalized_isic_id'] = df['patient_normalized_isic_id'] / df.groupby('patient_id')['patient_normalized_isic_id'].transform('max')
    df['patient_normalized_index'] = df.groupby('patient_id')['patient_id'].transform('cumcount')
    df['patient_normalized_index'] = df['patient_normalized_index'] / df.groupby('patient_id')['patient_normalized_index'].transform('max')

    df['patient_visit_id'] = df['age_approx'] - df.groupby('patient_id')['age_approx'].transform('min')
    df['patient_visit_count'] = df.groupby('patient_id')['patient_visit_id'].transform('nunique')
    df['patient_visit_id2'] = 0
    df.loc[(df['patient_visit_count'] == 2) & (df['patient_visit_id'] == 0), 'patient_visit_id2'] = 1
    df.loc[(df['patient_visit_count'] == 2) & (df['patient_visit_id'] == 5), 'patient_visit_id2'] = 2

    df['patient_anatom_site_general_nunique'] = df.groupby('patient_id')['anatom_site_general'].transform('nunique')
    df['patient_tbp_lv_location_nunique'] = df.groupby('patient_id')['tbp_lv_location'].transform('nunique')
    df['patient_tbp_lv_location_simple_nunique'] = df.groupby('patient_id')['tbp_lv_location_simple'].transform('nunique')

    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')
    df['patient_site_lesion_count'] = df.groupby(['patient_id', 'anatom_site_general'])['patient_id'].transform('count')
    df['patient_location_lesion_count'] = df.groupby(['patient_id', 'tbp_lv_location'])['patient_id'].transform('count')
    df['patient_location_simple_lesion_count'] = df.groupby(['patient_id', 'tbp_lv_location_simple'])['patient_id'].transform('count')

    df['patient_visit_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id'])['patient_id'].transform('count')
    df['patient_visit_site_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'anatom_site_general'])['patient_id'].transform('count')
    df['patient_visit_location_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'tbp_lv_location'])['patient_id'].transform('count')
    df['patient_visit_location_simple_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'tbp_lv_location_simple'])['patient_id'].transform('count')

    df['patient_site_lesion_count_ratio'] = df['patient_lesion_count'] / df['patient_site_lesion_count']
    df['patient_location_lesion_count_ratio'] = df['patient_lesion_count'] / df['patient_location_lesion_count']
    df['patient_location_simple_lesion_count_ratio'] = df['patient_lesion_count'] / df['patient_location_simple_lesion_count']

    aggregation_columns = [
        'tbp_lv_nevi_confidence',
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext',
        'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm',
        'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max', 'tbp_lv_norm_color',
        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
        'tbp_lv_area_perim_ratio', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_norm_border', 'tbp_lv_eccentricity',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',

        'image_model_prediction_rank'
    ]
    aggregations = ['mean', 'median', 'std', 'min', 'max', 'skew']

    df_patient_aggregations = df.groupby('patient_id').agg({column: aggregations for column in aggregation_columns})
    df_patient_aggregations.columns = 'patient_' + df_patient_aggregations.columns.map('_'.join).str.strip('_')
    df_patient_aggregations = df_patient_aggregations.reset_index()
    df = df.merge(df_patient_aggregations, on='patient_id', how='left')

    df_patient_site_aggregations = df.groupby(['patient_id', 'anatom_site_general']).agg({column: aggregations for column in aggregation_columns})
    df_patient_site_aggregations.columns = 'patient_site_' + df_patient_site_aggregations.columns.map('_'.join).str.strip('_')
    df_patient_site_aggregations = df_patient_site_aggregations.reset_index()
    df = df.merge(df_patient_site_aggregations, on=['patient_id', 'anatom_site_general'], how='left')

    df_patient_location_aggregations = df.groupby(['patient_id', 'tbp_lv_location']).agg({column: aggregations for column in aggregation_columns})
    df_patient_location_aggregations.columns = 'patient_location_' + df_patient_location_aggregations.columns.map('_'.join).str.strip('_')
    df_patient_location_aggregations = df_patient_location_aggregations.reset_index()
    df = df.merge(df_patient_location_aggregations, on=['patient_id', 'tbp_lv_location'], how='left')

    df_patient_location_simple_aggregations = df.groupby(['patient_id', 'tbp_lv_location_simple']).agg({column: aggregations for column in aggregation_columns})
    df_patient_location_simple_aggregations.columns = 'patient_location_simple_' + df_patient_location_simple_aggregations.columns.map('_'.join).str.strip('_')
    df_patient_location_simple_aggregations = df_patient_location_simple_aggregations.reset_index()
    df = df.merge(df_patient_location_simple_aggregations, on=['patient_id', 'tbp_lv_location_simple'], how='left')

    for column in aggregation_columns:

        df[f'patient_{column}_mean_ratio'] = df[column] / df[f'patient_{column}_mean']
        df[f'patient_{column}_median_ratio'] = df[column] / df[f'patient_{column}_median']
        df[f'patient_{column}_min_ratio'] = df[column] / df[f'patient_{column}_min']
        df[f'patient_{column}_max_ratio'] = df[column] / df[f'patient_{column}_max']
        df[f'patient_{column}_mean_difference'] = df[column] - df[f'patient_{column}_mean']
        df[f'patient_{column}_median_difference'] = df[column] - df[f'patient_{column}_median']
        df[f'patient_{column}_min_difference'] = df[column] - df[f'patient_{column}_min']
        df[f'patient_{column}_max_difference'] = df[column] - df[f'patient_{column}_max']

        df[f'patient_site_{column}_mean_ratio'] = df[column] / df[f'patient_site_{column}_mean']
        df[f'patient_site_{column}_median_ratio'] = df[column] / df[f'patient_site_{column}_median']
        df[f'patient_site_{column}_min_ratio'] = df[column] / df[f'patient_site_{column}_min']
        df[f'patient_site_{column}_max_ratio'] = df[column] / df[f'patient_site_{column}_max']
        df[f'patient_site_{column}_mean_difference'] = df[column] - df[f'patient_site_{column}_mean']
        df[f'patient_site_{column}_median_difference'] = df[column] - df[f'patient_site_{column}_median']
        df[f'patient_site_{column}_min_difference'] = df[column] - df[f'patient_site_{column}_min']
        df[f'patient_site_{column}_max_difference'] = df[column] - df[f'patient_site_{column}_max']

        df[f'patient_location_{column}_mean_ratio'] = df[column] / df[f'patient_location_{column}_mean']
        df[f'patient_location_{column}_median_ratio'] = df[column] / df[f'patient_location_{column}_median']
        df[f'patient_location_{column}_min_ratio'] = df[column] / df[f'patient_location_{column}_min']
        df[f'patient_location_{column}_max_ratio'] = df[column] / df[f'patient_location_{column}_max']
        df[f'patient_location_{column}_mean_difference'] = df[column] - df[f'patient_location_{column}_mean']
        df[f'patient_location_{column}_median_difference'] = df[column] - df[f'patient_location_{column}_median']
        df[f'patient_location_{column}_min_difference'] = df[column] - df[f'patient_location_{column}_min']
        df[f'patient_location_{column}_max_difference'] = df[column] - df[f'patient_location_{column}_max']

        df[f'patient_location_simple_{column}_mean_ratio'] = df[column] / df[f'patient_location_simple_{column}_mean']
        df[f'patient_location_simple_{column}_median_ratio'] = df[column] / df[f'patient_location_simple_{column}_median']
        df[f'patient_location_simple_{column}_min_ratio'] = df[column] / df[f'patient_location_simple_{column}_min']
        df[f'patient_location_simple_{column}_max_ratio'] = df[column] / df[f'patient_location_simple_{column}_max']
        df[f'patient_location_simple_{column}_mean_difference'] = df[column] - df[f'patient_location_simple_{column}_mean']
        df[f'patient_location_simple_{column}_median_difference'] = df[column] - df[f'patient_location_simple_{column}_median']
        df[f'patient_location_simple_{column}_min_difference'] = df[column] - df[f'patient_location_simple_{column}_min']
        df[f'patient_location_simple_{column}_max_difference'] = df[column] - df[f'patient_location_simple_{column}_max']

    return df


def create_rank_features(df):

    """
    Create rank features on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with raw features

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples)
        Dataframe with additional rank features
    """

    rank_columns = [
        'tbp_lv_nevi_confidence',
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext',
        'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm',
        'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max', 'tbp_lv_norm_color',
        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
        'tbp_lv_area_perim_ratio', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_norm_border', 'tbp_lv_eccentricity',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',

        'lesion_ab_ratio', 'lesion_ab_difference',
        'lesion_ac_ratio', 'lesion_ac_difference',
        'lesion_ah_ratio', 'lesion_ah_difference',
        'lesion_al_ratio', 'lesion_al_difference',
        'lesion_bc_ratio', 'lesion_bc_difference',
        'lesion_bh_ratio', 'lesion_bh_difference',
        'lesion_bl_ratio', 'lesion_bl_difference',
        'lesion_ch_ratio', 'lesion_ch_difference',
        'lesion_cl_ratio', 'lesion_cl_difference',
        'lesion_hl_ratio', 'lesion_hl_difference',

        'outside_ab_ratio', 'outside_ab_difference',
        'outside_ac_ratio', 'outside_ac_difference',
        'outside_ah_ratio', 'outside_ah_difference',
        'outside_al_ratio', 'outside_al_difference',
        'outside_bc_ratio', 'outside_bc_difference',
        'outside_bh_ratio', 'outside_bh_difference',
        'outside_bl_ratio', 'outside_bl_difference',
        'outside_ch_ratio', 'outside_ch_difference',
        'outside_cl_ratio', 'outside_cl_difference',
        'outside_hl_ratio', 'outside_hl_difference',

        'lesion_outside_ab_difference_ratio',
        'lesion_outside_ac_difference_ratio',
        'lesion_outside_ah_difference_ratio',
        'lesion_outside_al_difference_ratio',
        'lesion_outside_bc_difference_ratio',
        'lesion_outside_bh_difference_ratio',
        'lesion_outside_bl_difference_ratio',
        'lesion_outside_ch_difference_ratio',
        'lesion_outside_cl_difference_ratio',
        'lesion_outside_hl_difference_ratio',

        'a_lesion_outside_ratio', 'a_lesion_outside_difference',
        'b_lesion_outside_ratio', 'b_lesion_outside_difference',
        'c_lesion_outside_ratio', 'c_lesion_outside_difference',
        'l_lesion_outside_ratio', 'l_lesion_outside_difference',
        'h_lesion_outside_ratio', 'h_lesion_outside_difference',
        'stdl_lesion_outside_ratio', 'stdl_lesion_outside_difference',

        'delta_albnorm_ratio', 'delta_albnorm_difference',
        'delta_blbnorm_ratio', 'delta_blbnorm_difference',
        'delta_llbnorm_ratio', 'delta_llbnorm_difference',

        'lesion_diameter_ratio', 'lesion_diameter_difference', 'lesion_diameter_mean',
        'lesion_max_diameter_perimeter_ratio', 'lesion_max_diameter_perimeter_difference',
        'lesion_min_diameter_perimeter_ratio', 'lesion_min_diameter_perimeter_difference',
        'lesion_mean_diameter_perimeter_ratio', 'lesion_mean_diameter_perimeter_difference',
        'lesion_area_max_diameter_squared_ratio', 'lesion_area_max_diameter_squared_difference',
        'lesion_area_min_diameter_squared_ratio', 'lesion_area_min_diameter_squared_difference',
        'lesion_area_mean_diameter_squared_ratio', 'lesion_area_mean_diameter_squared_difference',
        'lesion_area_perimeter_squared_ratio', 'lesion_area_perimeter_squared_difference',
        'lesion_circularity', 'lesion_shape_index',

        'lesion_distance_to_origin',
        'lesion_angle_xy', 'lesion_angle_xz', 'lesion_angle_yz',

        'border_irregularity_mean', 'color_irregularity_mean',

        'image_model_prediction_rank'
    ] + [column for column in df.columns.tolist() if column.startswith('image_embedding')]
    df_patient_ranks = df.groupby('patient_id')[rank_columns].rank(pct=True).rename(
        columns={column: f'patient_{column}_rank' for column in rank_columns}
    )
    df_patient_site_ranks = df.groupby(['patient_id', 'anatom_site_general'])[rank_columns].rank(pct=True).rename(
        columns={column: f'patient_site_{column}_rank' for column in rank_columns}
    )
    df_patient_location_ranks = df.groupby(['patient_id', 'tbp_lv_location'])[rank_columns].rank(pct=True).rename(
        columns={column: f'patient_location_{column}_rank' for column in rank_columns}
    )
    df_patient_location_simple_ranks = df.groupby(['patient_id', 'tbp_lv_location_simple'])[rank_columns].rank(pct=True).rename(
        columns={column: f'patient_location_simple_{column}_rank' for column in rank_columns}
    )
    df = pd.concat((
        df, df_patient_ranks, df_patient_site_ranks, df_patient_location_ranks, df_patient_location_simple_ranks
    ), axis=1, ignore_index=False)

    return df


def create_anchor_features(df):

    anchor_columns = ['clin_size_long_diam_mm', 'image_model_prediction_rank']
    feature_columns = [
        'tbp_lv_nevi_confidence',
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext',
        'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm',
        'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max', 'tbp_lv_norm_color',
        'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
        'tbp_lv_area_perim_ratio', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_norm_border', 'tbp_lv_eccentricity',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',

        'image_model_prediction_rank'
    ]

    for anchor_column in anchor_columns:

        df_anchor_max = df.loc[df.groupby('patient_id')[anchor_column].idxmax(), ['patient_id'] + feature_columns].rename(columns={
            column: f'{column}_{anchor_column}_anchor_max' for column in feature_columns
        })
        df = df.merge(df_anchor_max, on='patient_id', how='left')

        df_anchor_min = df.loc[df.groupby('patient_id')[anchor_column].idxmin(), ['patient_id'] + feature_columns].rename(columns={
            column: f'{column}_{anchor_column}_anchor_min' for column in feature_columns
        })
        df = df.merge(df_anchor_min, on='patient_id', how='left')

        for feature_column in feature_columns:
            df[f'{feature_column}_{anchor_column}_anchor_max_ratio'] = df[feature_column] / df[f'{feature_column}_{anchor_column}_anchor_max']
            df[f'{feature_column}_{anchor_column}_anchor_max_difference'] = df[feature_column] - df[f'{feature_column}_{anchor_column}_anchor_max']

            df[f'{feature_column}_{anchor_column}_anchor_min_ratio'] = df[feature_column] / df[f'{feature_column}_{anchor_column}_anchor_min']
            df[f'{feature_column}_{anchor_column}_anchor_min_difference'] = df[feature_column] - df[f'{feature_column}_{anchor_column}_anchor_min']

    return df


def preprocess(df, categorical_columns, encoder_directory, data_directory):

    df = create_multiclass_target(df=df)

    df = create_color_features(df=df)
    df = create_shape_features(df=df)
    df = create_coordinate_features(df=df)
    df = create_irregularity_features(df=df)

    df = pd.concat((
        df,
        pd.read_parquet(data_directory / 'patient_group_features.parquet')
    ), axis=1, ignore_index=False)

    df = pd.concat((
        df,
        pd.read_parquet(data_directory / 'patient_distance_features.parquet')
    ), axis=1, ignore_index=False)

    df = pd.concat((
        df,
        pd.read_parquet(data_directory / 'image_embeddings_features.parquet')
    ), axis=1, ignore_index=False)

    df = pd.concat((
        df,
        pd.read_parquet(data_directory / 'image_distance_features.parquet')
    ), axis=1, ignore_index=False)

    df = df.merge(
        pd.read_parquet(data_directory / 'oof_predictions.parquet')[['isic_id', 'prediction', 'prediction_rank']].rename(columns={
            'prediction': 'image_model_prediction',
            'prediction_rank': 'image_model_prediction_rank'
        }),
        on='isic_id',
        how='left'
    )

    df = encode_categorical_columns(df=df, categorical_columns=categorical_columns, encoder_directory=encoder_directory)
    df = create_aggregation_features(df=df)
    df = create_rank_features(df=df)
    df = create_anchor_features(df=df)

    return df
