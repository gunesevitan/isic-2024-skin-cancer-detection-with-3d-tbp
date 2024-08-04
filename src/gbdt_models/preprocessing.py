import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def encode_categorical_columns(df, categorical_columns):

    encoders = {}
    for column in categorical_columns:
        encoder = OrdinalEncoder(
            categories='auto',
            dtype=int,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        df[column] = encoder.fit_transform(df[column].values.reshape(-1, 1))
        encoders[column] = encoder

    return df


def create_interaction_features(df):

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



    df['a_ratio'] = df['tbp_lv_A'] / df['tbp_lv_Aext']
    df['a_difference'] = df['tbp_lv_A'] - df['tbp_lv_Aext']
    df['b_ratio'] = df['tbp_lv_B'] / df['tbp_lv_Bext']
    df['b_difference'] = df['tbp_lv_B'] - df['tbp_lv_Bext']
    df['c_ratio'] = df['tbp_lv_C'] / df['tbp_lv_Cext']
    df['c_difference'] = df['tbp_lv_C'] - df['tbp_lv_Cext']
    df['h_ratio'] = df['tbp_lv_H'] / df['tbp_lv_Hext']
    df['h_difference'] = df['tbp_lv_H'] - df['tbp_lv_Hext']
    df['l_ratio'] = df['tbp_lv_L'] / df['tbp_lv_Lext']
    df['l_difference'] = df['tbp_lv_L'] - df['tbp_lv_Lext']

    return df


def create_patient_aggregations(df):

    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')

    return df

def preprocess(df, categorical_columns):

    df = encode_categorical_columns(df=df, categorical_columns=categorical_columns)
    df = create_interaction_features(df=df)
    df = create_patient_aggregations(df=df)


    return df
