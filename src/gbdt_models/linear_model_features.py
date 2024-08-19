import sys
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append('..')
import settings
import preprocessing
import metrics


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df = preprocessing.create_color_features(df=df)

    column_groups = [
        ['tbp_lv_A', 'tbp_lv_B', 'tbp_lv_L'],
        ['tbp_lv_H', 'tbp_lv_C'],
        ['tbp_lv_A', 'tbp_lv_B', 'tbp_lv_C', 'tbp_lv_H', 'tbp_lv_L'],

        ['tbp_lv_Aext', 'tbp_lv_Bext', 'tbp_lv_Lext'],
        ['tbp_lv_Hext', 'tbp_lv_Cext'],
        ['tbp_lv_Aext', 'tbp_lv_Bext', 'tbp_lv_Cext', 'tbp_lv_Hext', 'tbp_lv_Lext'],

        ['tbp_lv_A', 'tbp_lv_B', 'tbp_lv_C', 'tbp_lv_H', 'tbp_lv_L', 'tbp_lv_Aext', 'tbp_lv_Bext', 'tbp_lv_Cext', 'tbp_lv_Hext', 'tbp_lv_Lext'],

        ['lesion_x', 'lesion_y', 'lesion_z'],
        ['lesion_r', 'lesion_g', 'lesion_b'],
        ['lesion_h', 'lesion_s', 'lesion_v'],
    ]

    for column_group in column_groups:
        model = LinearRegression()
        model.fit(
            df[column_group] / 1e4,
            df['target']
        )
        predictions = model.predict(df[column_group])
        scores = metrics.classification_scores(
            y_true=df.loc[:, 'target'],
            y_pred=predictions,
        )
        coefficients = {feature: coefficient for feature, coefficient in zip(column_group, model.coef_)}
        settings.logger.info(
            f'''
            Column Group {column_group}
            Scores {json.dumps(scores, indent=2)}
            Coefficients {json.dumps(coefficients, indent=2)}
            '''
        )
