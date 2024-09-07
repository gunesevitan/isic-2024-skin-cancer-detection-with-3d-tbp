import sys
import warnings
import yaml
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

sys.path.append('..')
import settings
import preprocessing
import metrics


def load_model(model_directory):

    models = {}

    for model_path in tqdm(sorted(list(model_directory.glob('model*')))):

        model_path = str(model_path)

        if 'lightgbm' in model_path:
            model = lgb.Booster(model_file=model_path)
        elif 'xgboost' in model_path:
            model = xgb.Booster()
            model.load_model(model_path)
        elif 'catboost' in model_path:
            model = cb.CatBoostRegressor()
            model.load_model(model_path)
        else:
            raise ValueError('Invalid model type')

        models[model_path.split('/')[-1].split('.')[0]] = model

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    return config, models


def predict_train(df, config, models, model_type, task_type, prediction_iteration_quantiles, prediction_column_name, verbose=False):

    scores = []

    if model_type == 'lightgbm' or model_type == 'xgboost':
        total_iterations = config['fit_parameters']['boosting_rounds']
    elif model_type == 'catboost':
        total_iterations = config['model_parameters']['iterations']
    else:
        raise ValueError('Invalid model type')

    prediction_iterations = np.quantile(np.arange(1, total_iterations + 1), q=prediction_iteration_quantiles).astype(int)
    seeds = sorted(list(set([seed.split('_')[-1] for seed in models.keys()])))
    folds = sorted(list(set([seed.split('_')[2] for seed in models.keys()])))

    for fold in folds:

        validation_mask = df[f'fold{fold}'] == 1
        df.loc[validation_mask, prediction_column_name] = 0

        for seed in seeds:

            model = models[f'model_fold_{fold}_seed_{seed}']

            for iteration in prediction_iterations:

                if model_type == 'lightgbm':
                    validation_predictions = model.predict(
                        df.loc[validation_mask, config['training']['features']],
                        start_iteration=0,
                        num_iteration=int(iteration)
                    )
                elif model_type == 'xgboost':
                    validation_predictions = model.predict(
                        xgb.DMatrix(df.loc[validation_mask, config['training']['features']].replace([np.inf, -np.inf], 0)),
                        iteration_range=(0, int(iteration))
                    )
                elif model_type == 'catboost':
                    validation_predictions = model.predict(
                        df.loc[validation_mask, config['training']['features']],
                        ntree_start=0,
                        ntree_end=iteration
                    )
                else:
                    raise ValueError('Invalid model type')

                if task_type == 'multiclass':
                    validation_predictions = validation_predictions[:, 1]

                df.loc[validation_mask, prediction_column_name] += (pd.Series(validation_predictions).rank(pct=True).values / (len(prediction_iterations) * len(seeds)))

                if verbose:
                    settings.logger.info(
                        f'''
                        Predicted with {model_type} model seed {seed} iteration {iteration}
                        Prediction Mean: {np.mean(validation_predictions):.2f} Std: {np.std(validation_predictions):.2f} Min: {np.min(validation_predictions):.2f} Max: {np.max(validation_predictions):.2f}
                        '''
                    )

        validation_scores = metrics.classification_scores(
            y_true=df.loc[validation_mask, 'target'],
            y_pred=df.loc[validation_mask, prediction_column_name],
        )
        scores.append(validation_scores)

        if verbose:
            settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(validation_scores, indent=2)}')

    scores = pd.DataFrame(scores)
    if verbose:
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
            '''
        )
    oof_mask = df[prediction_column_name].notna()
    oof_scores = metrics.classification_scores(
        y_true=df.loc[oof_mask, 'target'],
        y_pred=df.loc[oof_mask, prediction_column_name],
    )
    if verbose:
        settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    return df


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    df = df.merge(pd.read_csv(settings.DATA / 'folds.csv'), on='isic_id', how='left')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=[
            'sex', 'anatom_site_general', 'tbp_tile_type',
            'tbp_lv_location', 'tbp_lv_location_simple',
            'attribution'
        ],
        encoder_directory=settings.MODELS / 'encoders',
        data_directory=settings.DATA
    )

    lightgbm_tabular_binary_config, lightgbm_tabular_binary_models = load_model(
        model_directory=settings.MODELS / 'lightgbm_tabular_binary'
    )

    lightgbm_tabular_multiclass_config, lightgbm_tabular_multiclass_models = load_model(
        model_directory=settings.MODELS / 'lightgbm_tabular_multiclass'
    )

    lightgbm_tabular_image_binary_config, lightgbm_tabular_image_binary_models = load_model(
        model_directory=settings.MODELS / 'lightgbm_tabular_image_binary'
    )

    lightgbm_tabular_image_multiclass_config, lightgbm_tabular_image_multiclass_models = load_model(
        model_directory=settings.MODELS / 'lightgbm_tabular_image_multiclass'
    )

    df = predict_train(
        df=df,
        config=lightgbm_tabular_binary_config,
        models=lightgbm_tabular_binary_models,
        model_type='lightgbm',
        task_type='binary',
        prediction_iteration_quantiles=(1,),
        prediction_column_name='lightgbm_tabular_binary_prediction',
        verbose=True
    )
    df = predict_train(
        df=df,
        config=lightgbm_tabular_multiclass_config,
        models=lightgbm_tabular_multiclass_models,
        model_type='lightgbm',
        task_type='multiclass',
        prediction_iteration_quantiles=(1,),
        prediction_column_name='lightgbm_tabular_multiclass_prediction',
        verbose=True
    )
    df = predict_train(
        df=df,
        config=lightgbm_tabular_image_binary_config,
        models=lightgbm_tabular_image_binary_models,
        model_type='lightgbm',
        task_type='binary',
        prediction_iteration_quantiles=(1,),
        prediction_column_name='lightgbm_tabular_image_binary_prediction',
        verbose=True
    )
    df = predict_train(
        df=df,
        config=lightgbm_tabular_image_multiclass_config,
        models=lightgbm_tabular_image_multiclass_models,
        model_type='lightgbm',
        task_type='multiclass',
        prediction_iteration_quantiles=(1,),
        prediction_column_name='lightgbm_tabular_image_multiclass_prediction',
        verbose=True
    )

    df['lightgbm_tabular_prediction'] = (df['lightgbm_tabular_binary_prediction'] * 0.5) + \
                                        (df['lightgbm_tabular_multiclass_prediction'] * 0.5)

    df['lightgbm_tabular_image_prediction'] = (df['lightgbm_tabular_image_binary_prediction'] * 0.5) + \
                                              (df['lightgbm_tabular_image_multiclass_prediction'] * 0.5)

    df['lightgbm_prediction'] = (df['lightgbm_tabular_prediction'] * 0.15) + \
                                (df['lightgbm_tabular_image_prediction'] * 0.85)

    oof_mask = df['blend'].notna()
    oof_scores = metrics.classification_scores(
        y_true=df.loc[oof_mask, 'target'],
        y_pred=df.loc[oof_mask, 'lightgbm_prediction'],
    )
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')