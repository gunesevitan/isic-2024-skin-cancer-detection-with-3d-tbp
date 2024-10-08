import sys
import argparse
import warnings
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import catboost as cb

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    df = df.merge(pd.read_csv(settings.DATA / 'folds.csv'), on='isic_id', how='left')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=config['preprocessing']['categorical_columns'],
        encoder_directory=model_directory
    )

    task_type = config['training']['task_type']
    target = config['training']['target']
    features = config['training']['features']
    categorical_features = config['training']['categorical_features']
    folds = config['training']['folds']
    seeds = config['training']['seeds']

    settings.logger.info(
        f'''
        Running CatBoost trainer in {args.mode} mode ({task_type} task type)
        Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Categorical Features: {json.dumps(categorical_features, indent=2)}
        Target: {target}
        '''
    )

    if args.mode == 'validation':

        df_feature_importance_gain = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        scores = []
        curves = []

        for fold in folds:

            training_mask = (df[f'fold{fold}'] == 0) & (df['tbp_lv_dnn_lesion_confidence'] > 0.0)
            validation_mask = df[f'fold{fold}'] == 1

            settings.logger.info(
                f'''
                Fold: {fold} 
                Training: ({training_mask.sum()}) - Target Mean: {df.loc[training_mask, target].mean():.4f}
                Validation: ({validation_mask.sum()}) - Target Mean: {df.loc[validation_mask, target].mean():.4f}
                '''
            )

            df.loc[validation_mask, 'prediction'] = 0

            for seed in seeds:

                training_dataset = cb.Pool(
                    df.loc[training_mask, features],
                    label=df.loc[training_mask, target],
                    cat_features=categorical_features
                )
                validation_dataset = cb.Pool(
                    df.loc[validation_mask, features],
                    label=df.loc[validation_mask, target],
                    cat_features=categorical_features
                )

                config['model_parameters']['random_seed'] = seed

                model = cb.train(
                    params=config['model_parameters'],
                    dtrain=training_dataset,
                    evals=[training_dataset, validation_dataset]
                )
                model.save_model(model_directory / f'model_fold_{fold}_seed_{seed}.cbm')

                df_feature_importance_gain[fold] += model.get_feature_importance()
                validation_predictions = model.predict(df.loc[validation_mask, features])

                if task_type == 'multiclass':
                    validation_predictions = validation_predictions[:, 1]

                df.loc[validation_mask, 'prediction'] += (pd.Series(validation_predictions).rank(pct=True).values / len(seeds))

            validation_scores = metrics.classification_scores(
                y_true=df.loc[validation_mask, 'target'],
                y_pred=df.loc[validation_mask, 'prediction'],
            )
            settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(validation_scores, indent=2)}')
            scores.append(validation_scores)

            validation_curves = metrics.classification_curves(
                y_true=df.loc[validation_mask, 'target'],
                y_pred=df.loc[validation_mask, 'prediction'],
            )
            curves.append(validation_curves)

        scores = pd.DataFrame(scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        oof_mask = df['prediction'].notna()
        oof_scores = metrics.classification_scores(
            y_true=df.loc[oof_mask, 'target'],
            y_pred=df.loc[oof_mask, 'prediction'],
        )
        settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

        scores = pd.concat((
            scores,
            pd.DataFrame([oof_scores])
        )).reset_index(drop=True)
        scores['fold'] = folds + ['OOF']
        scores = scores[scores.columns.tolist()[::-1]]
        scores.to_csv(model_directory / 'scores.csv', index=False)
        settings.logger.info(f'scores.csv is saved to {model_directory}')

        visualization.visualize_scores(
            scores=scores,
            title=f'CatBoost Model Scores of {len(folds)} Fold(s)',
            path=model_directory / 'scores.png'
        )
        settings.logger.info(f'Saved scores.png to {model_directory}')

        visualization.visualize_roc_curves(
            roc_curves=[curve['roc'] for curve in curves],
            title='CatBoost Validation ROC Curves',
            path=model_directory / 'roc_curves.png'
        )

        visualization.visualize_pr_curves(
            pr_curves=[curve['pr'] for curve in curves],
            title='CatBoost Validation PR Curves',
            path=model_directory / 'pr_curves.png'
        )

        visualization.visualize_predictions(
            y_true=df['target'],
            y_pred=df['prediction'],
            title='CatBoost Predictions Histogram',
            path=model_directory / 'predictions.png'
        )

        for importance_type, df_feature_importance in zip(['gain'], [df_feature_importance_gain]):
            df_feature_importance['mean'] = df_feature_importance[config['training']['folds']].mean(axis=1)
            df_feature_importance['std'] = df_feature_importance[config['training']['folds']].std(axis=1).fillna(0)
            df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)
            visualization.visualize_feature_importance(
                df_feature_importance=df_feature_importance,
                title=f'CatBoost Feature Importance ({importance_type.capitalize()})',
                path=model_directory / f'feature_importance_{importance_type}.png'
            )
            settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

        df.loc[:, [target, 'prediction']].to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'Saved oof_predictions.csv to {model_directory}')

    elif args.mode == 'submission':

        for seed in seeds:

            training_dataset = cb.Pool(df.loc[:, features], label=df.loc[:, target], cat_features=features)

            config['model_parameters']['random_seed'] = seed

            model = cb.train(
                params=config['model_parameters'],
                dtrain=training_dataset,
                evals=[training_dataset]
            )
            model.save_model(model_directory / f'model_{seed}.cbm')

    else:
        raise ValueError(f'Invalid mode {args.mode}')
