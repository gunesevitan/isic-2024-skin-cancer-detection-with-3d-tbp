import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import feature_utilities


def encode_categorical_columns(df, categorical_columns):

    encoders = {}
    for column in categorical_columns:
        encoder = OrdinalEncoder(
            categories='auto',
            dtype=np.float32,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        df[column] = encoder.fit_transform(df[column].values.reshape(-1, 1))
        encoders[column] = encoder

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

    lesion_abchl_ratio_columns = [
        'lesion_ab_ratio', 'lesion_ac_ratio', 'lesion_ah_ratio', 'lesion_al_ratio',
        'lesion_bc_ratio', 'lesion_bh_ratio', 'lesion_bl_ratio',
        'lesion_ch_ratio', 'lesion_cl_ratio',
        'lesion_hl_ratio'
    ]
    # LAB space variable interactions for lesion aggregations
    df['lesion_abchl_ratio_mean'] = df[lesion_abchl_ratio_columns].mean(axis=1)
    df['lesion_abchl_ratio_std'] = df[lesion_abchl_ratio_columns].std(axis=1)
    df['lesion_abchl_ratio_min'] = df[lesion_abchl_ratio_columns].min(axis=1)
    df['lesion_abchl_ratio_max'] = df[lesion_abchl_ratio_columns].max(axis=1)
    df['lesion_abchl_ratio_sum'] = df[lesion_abchl_ratio_columns].sum(axis=1)
    df['lesion_abchl_ratio_skew'] = df[lesion_abchl_ratio_columns].skew(axis=1)

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

    outside_abchl_ratio_columns = [
        'outside_ab_ratio', 'outside_ac_ratio', 'outside_ah_ratio', 'outside_al_ratio',
        'outside_bc_ratio', 'outside_bh_ratio', 'outside_bl_ratio',
        'outside_ch_ratio', 'outside_cl_ratio',
        'outside_hl_ratio'
    ]
    # LAB space variable interactions for outside aggregations
    df['outside_abchl_ratio_mean'] = df[outside_abchl_ratio_columns].mean(axis=1)
    df['outside_abchl_ratio_std'] = df[outside_abchl_ratio_columns].std(axis=1)
    df['outside_abchl_ratio_min'] = df[outside_abchl_ratio_columns].min(axis=1)
    df['outside_abchl_ratio_max'] = df[outside_abchl_ratio_columns].max(axis=1)
    df['outside_abchl_ratio_sum'] = df[outside_abchl_ratio_columns].sum(axis=1)
    df['outside_abchl_ratio_skew'] = df[outside_abchl_ratio_columns].skew(axis=1)

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
    color_comparison_columns = [
        ('tbp_lv_A', 'tbp_lv_Aext'),
        ('tbp_lv_B', 'tbp_lv_Bext'),
        ('tbp_lv_C', 'tbp_lv_Cext'),
        ('tbp_lv_H', 'tbp_lv_Hext'),
        ('tbp_lv_L', 'tbp_lv_Lext'),
    ]
    for lesion_color_column, outside_color_column in color_comparison_columns:
        column_name = lesion_color_column.split('_')[-1].lower()
        df[f'{column_name}_lesion_outside_ratio'] = df[lesion_color_column] / df[outside_color_column]
        df[f'{column_name}_lesion_outside_difference'] = df[lesion_color_column] - df[outside_color_column]

    return df


def create_shape_features(df):

    pass


def create_interaction_features(df):

    # Different shape areas and perimeters
    df['lesion_rectangle_area'] = df['clin_size_long_diam_mm'] * df['tbp_lv_minorAxisMM']
    df['lesion_rectangle_perimeter'] = 2 * (df['clin_size_long_diam_mm'] + df['tbp_lv_minorAxisMM'])
    df['lesion_ellipse_area'] = np.pi * (df['clin_size_long_diam_mm'] / 2) * (df['tbp_lv_minorAxisMM'] / 2)
    df['lesion_ellipse_perimeter'] = np.pi * np.sqrt(2 * (df['clin_size_long_diam_mm'] / 2) ** 2 + 2 * (df['tbp_lv_minorAxisMM'] / 2) ** 2)

    df['lesion_diameter_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_minorAxisMM']
    df['lesion_diameter_difference'] = df['clin_size_long_diam_mm'] - df['tbp_lv_minorAxisMM']
    df['lesion_diamater_mean'] = (df['clin_size_long_diam_mm'] + df['tbp_lv_minorAxisMM']) / 2

    df['lesion_area_max_diameter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['clin_size_long_diam_mm'] ** 2)
    df['lesion_area_max_diameter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['clin_size_long_diam_mm'] ** 2)
    df['lesion_area_min_diameter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_minorAxisMM'] ** 2)
    df['lesion_area_min_diameter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['tbp_lv_minorAxisMM'] ** 2)
    df['lesion_area_perimeter_squared_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2)
    df['lesion_area_perimeter_squared_difference'] = df['tbp_lv_areaMM2'] - (df['tbp_lv_perimeterMM'] ** 2)

    df['lesion_max_diameter_perimeter_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_perimeterMM']
    df['lesion_max_diameter_perimeter_difference'] = df['clin_size_long_diam_mm'] - df['tbp_lv_perimeterMM']
    df['lesion_min_diameter_perimeter_ratio'] = df['tbp_lv_minorAxisMM'] / df['tbp_lv_perimeterMM']
    df['lesion_min_diameter_perimeter_difference'] = df['tbp_lv_minorAxisMM'] - df['tbp_lv_perimeterMM']

    df['lesion_circularity'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM'] ** 2)
    df['lesion_shape_index'] = df['tbp_lv_perimeterMM'] / np.sqrt(df['tbp_lv_areaMM2'])


    df['delta_albnorm_ratio'] = df['tbp_lv_deltaA'] / df['tbp_lv_deltaLBnorm']
    df['delta_albnorm_difference'] = df['tbp_lv_deltaA'] - df['tbp_lv_deltaLBnorm']
    df['delta_blbnorm_ratio'] = df['tbp_lv_deltaB'] / df['tbp_lv_deltaLBnorm']
    df['delta_blbnorm_difference'] = df['tbp_lv_deltaB'] - df['tbp_lv_deltaLBnorm']
    df['delta_llbnorm_ratio'] = df['tbp_lv_deltaL'] / df['tbp_lv_deltaLBnorm']
    df['delta_llbnorm_difference'] = df['tbp_lv_deltaL'] - df['tbp_lv_deltaLBnorm']

    border_irregularity_columns = ['tbp_lv_norm_border', 'tbp_lv_symm_2axis']
    df['border_irregularity_mean'] = df[border_irregularity_columns].mean(axis=1)
    df['border_irregularity_std'] = df[border_irregularity_columns].std(axis=1)
    df['border_irregularity_min'] = df[border_irregularity_columns].min(axis=1)
    df['border_irregularity_max'] = df[border_irregularity_columns].max(axis=1)
    df['border_irregularity_sum'] = df[border_irregularity_columns].sum(axis=1)
    df['border_irregularity_skew'] = df[border_irregularity_columns].skew(axis=1)
    df['border_irregularity_ratio'] = df['tbp_lv_norm_border'] / df['tbp_lv_symm_2axis']
    df['border_irregularity_difference'] = df['tbp_lv_norm_border'] - df['tbp_lv_symm_2axis']

    color_irregularity_columns = ['tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max']
    df['color_irregularity_mean'] = df[color_irregularity_columns].mean(axis=1)
    df['color_irregularity_std'] = df[color_irregularity_columns].std(axis=1)
    df['color_irregularity_min'] = df[color_irregularity_columns].min(axis=1)
    df['color_irregularity_max'] = df[color_irregularity_columns].max(axis=1)
    df['color_irregularity_sum'] = df[color_irregularity_columns].sum(axis=1)
    df['color_irregularity_skew'] = df[color_irregularity_columns].skew(axis=1)
    df['color_irregularity_ratio'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_radial_color_std_max']
    df['color_irregularity_difference'] = df['tbp_lv_color_std_mean'] - df['tbp_lv_radial_color_std_max']

    irregularity_columns = border_irregularity_columns + color_irregularity_columns
    df['irregularity_mean'] = df[irregularity_columns].mean(axis=1)
    df['irregularity_std'] = df[irregularity_columns].std(axis=1)
    df['irregularity_min'] = df[irregularity_columns].min(axis=1)
    df['irregularity_max'] = df[irregularity_columns].max(axis=1)
    df['irregularity_sum'] = df[irregularity_columns].sum(axis=1)
    df['irregularity_skew'] = df[irregularity_columns].skew(axis=1)

    df['distance_to_origin'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
    df['angle_xy'] = np.arctan2(df['tbp_lv_x'], df['tbp_lv_y'])
    df['angle_xz'] = np.arctan2(df['tbp_lv_x'], df['tbp_lv_z'])
    df['angle_yz'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_z'])

    df['site_location_id'] = df['anatom_site_general'].astype(str) + '_' + df['tbp_lv_location'].astype(str)

    return df


def create_patient_aggregations(df):

    df['patient_id_count'] = df.groupby('patient_id')['patient_id'].transform('count')

    df_patient_aggregations = df.groupby('patient_id').agg({
        'clin_size_long_diam_mm': ['mean', 'median', 'std', 'min', 'max', 'sum', 'skew']
    })
    df_patient_aggregations.columns = 'patient_id_' + df_patient_aggregations.columns.map('_'.join).str.strip('_')
    df_patient_aggregations = df_patient_aggregations.reset_index()
    df = df.merge(df_patient_aggregations, on='patient_id', how='left')

    patient_id_rank_columns = [
        'clin_size_long_diam_mm',
        'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext',
        'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm',
        'tbp_lv_eccentricity',
        'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',

        'lesion_diameter_ratio', 'lesion_diameter_difference', 'lesion_diamater_mean',
        'lesion_area_max_diameter_squared_ratio', 'lesion_area_max_diameter_squared_difference',
        'lesion_area_min_diameter_squared_ratio', 'lesion_area_min_diameter_squared_difference',
        'lesion_area_perimeter_squared_ratio', 'lesion_area_perimeter_squared_difference',
    ]
    df_patient_id_ranks = df.groupby('patient_id')[patient_id_rank_columns].rank(pct=True).rename(
        columns={column: f'patient_id_{column}_rank' for column in patient_id_rank_columns}
    )
    df = pd.concat((df, df_patient_id_ranks), axis=1, ignore_index=False)

    return df

def preprocess(df, categorical_columns):

    df = create_color_features(df=df)
    df = create_interaction_features(df=df)
    df = encode_categorical_columns(df=df, categorical_columns=categorical_columns)
    df = create_patient_aggregations(df=df)


    return df
