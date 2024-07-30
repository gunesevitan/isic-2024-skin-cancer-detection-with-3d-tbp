import os
import sys
import warnings
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import torch_datasets
import torch_modules
import torch_utilities
import transforms
import metrics
import visualization
sys.path.append('..')
import settings


def train(training_loader, model, criterion, optimizer, device, scheduler=None, amp=False):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    training_outputs: dict
        Dictionary of training outputs
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss = 0.0

    if amp:
        grad_scaler = torch.amp.GradScaler(device=device)
    else:
        grad_scaler = None

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets.float())

        if amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_loss = running_loss / len(training_loader.sampler)

    training_outputs = {
        'training_loss': training_loss
    }

    return training_outputs


def validate(validation_loader, model, criterion, device, amp=False):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    validation_outputs: dict
        Dictionary of validation outputs

    validation_predictions: torch.Tensor of shape (batch_size, n_outputs)
        Validation predictions
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

        loss = criterion(outputs, targets.float())

        running_loss += loss.detach().item() * len(inputs)
        validation_predictions.append(outputs.detach().cpu())
        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_predictions = torch.cat(validation_predictions, dim=0).float()

    validation_outputs = {
        'validation_loss': validation_loss,
    }

    return validation_outputs, validation_predictions


if __name__ == '__main__':

    #warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    df = pd.read_parquet(settings.DATA / 'isic_master_dataset' / 'metadata.parquet').merge(
        pd.read_csv(settings.DATA / 'folds.csv'),
        on='isic_id',
        how='right'
    )

    settings.logger.info(f'Dataset Shape {df.shape}')

    image_paths, features, targets = torch_datasets.prepare_dataset(df=df, features=None)
    image_transforms = transforms.get_image_transforms(**config['transforms'])
    training_datasets = config['dataset']['training']
    validation_datasets = config['dataset']['validation']

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        training_metadata = {}
        training_folds = config['training']['folds']

        for fold in training_folds:

            training_mask = (df[f'fold{fold}'] == 0) & (df['dataset'].isin(training_datasets))
            validation_mask = (df[f'fold{fold}'] == 1) & (df['dataset'].isin(validation_datasets))

            np.random.seed(42)
            training_positive_idx = np.where(training_mask & (df['target'] == 1))[0]
            training_negative_idx = np.random.choice(np.where(training_mask & (df['target'] == 0))[0], 10000)
            training_idx = np.concatenate((training_positive_idx, training_negative_idx))

            settings.logger.info(
                f'''
                Fold {fold}
                Training Size {training_idx.shape[0]} ({training_idx.shape[0] // config["training"]["training_batch_size"] + 1} steps)
                Validation Size: {np.sum(validation_mask)} ({np.sum(validation_mask) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = torch_datasets.TabularImageDataset(
                image_paths=image_paths[training_idx],
                features=None,
                targets=targets[training_idx],
                transforms=image_transforms['training']
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )
            validation_dataset = torch_datasets.TabularImageDataset(
                image_paths=image_paths[validation_mask],
                features=None,
                targets=targets[validation_mask],
                transforms=image_transforms['inference']
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model_checkpoint_path = config['model']['model_checkpoint_path']
            if model_checkpoint_path is not None:
                model_checkpoint_path = settings.MODELS / model_checkpoint_path
                model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            early_stopping_patience = config['training']['early_stopping_patience']
            early_stopping_metric = config['training']['early_stopping_metric']
            training_history = {f'{dataset}_{metric}': [] for metric in config['persistence']['save_best_metrics'] for dataset in ['training', 'validation']}

            for epoch in range(1, config['training']['epochs'] + 1):

                training_outputs = train(
                    training_loader=training_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_outputs, validation_predictions = validate(
                    validation_loader=validation_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    amp=amp
                )
                validation_predictions = torch.sigmoid(validation_predictions).numpy()
                df.loc[validation_mask, 'prediction'] = validation_predictions

                training_results = {k.replace('training_', ''): v for k, v in training_outputs.items() if 'loss' in k}
                validation_results = {k.replace('validation_', ''): v for k, v in validation_outputs.items() if 'loss' in k}

                if len(validation_datasets) > 1:
                    for dataset in validation_datasets:
                        dataset_mask = validation_mask & (df['dataset'] == dataset)
                        dataset_validation_scores = metrics.classification_scores(
                            y_true=df.loc[dataset_mask, 'target'],
                            y_pred=df.loc[dataset_mask, 'prediction'],
                        )
                        dataset_validation_scores = {f'{dataset}_{k}': v for k, v in dataset_validation_scores.items()}
                        validation_results.update(dataset_validation_scores)

                validation_scores = metrics.classification_scores(
                    y_true=df.loc[validation_mask, 'target'],
                    y_pred=df.loc[validation_mask, 'prediction'],
                )
                validation_results.update(validation_scores)

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Loss: {json.dumps(training_results, indent=2)}
                    Validation Loss: {json.dumps(validation_results, indent=2)}
                    '''
                )

                if epoch in config['persistence']['save_epochs']:
                    # Save model if epoch is specified to be saved
                    model_name = f'model_fold_{fold}_epoch_{epoch}.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory}')

                for metric in config['persistence']['save_best_metrics']:

                    best_validation_metric = np.min(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else np.inf
                    last_validation_metric = validation_results[metric]
                    
                    if last_validation_metric < best_validation_metric:

                        previous_model = glob(str(model_directory / f'model_fold_{fold}_epoch_*_best_{metric}*'))
                        if len(previous_model) > 0:
                            os.remove(previous_model[0])
                            settings.logger.info(f'Deleted {previous_model[0].split("/")[-1]} from {model_directory}')

                        # Save model if specified validation metric improves
                        model_name = f'model_fold_{fold}_epoch_{epoch}_best_{metric}_{last_validation_metric:.6f}.pt'
                        torch.save(model.state_dict(), model_directory / model_name)
                        settings.logger.info(f'Saved {model_name} to {model_directory} (validation {metric} decreased from {best_validation_metric:.6f} to {last_validation_metric:.6f})\n')

                    if metric == 'loss':
                        training_history[f'training_{metric}'].append(training_results[metric])
                    training_history[f'validation_{metric}'].append(validation_results[metric])

            training_metadata['training_history'] = training_history
            for metric in config['persistence']['save_best_metrics']:
                best_epoch = int(np.argmin(training_history[f'validation_{metric}']))
                training_metadata[f'best_epoch_{metric}'] = best_epoch + 1
                training_metadata[f'training_{metric}'] = float(training_history[f'training_{metric}'][best_epoch])
                training_metadata[f'validation_{metric}'] = float(training_history[f'validation_{metric}'][best_epoch])
                visualization.visualize_learning_curve(
                    training_scores=training_metadata['training_history'][f'training_{metric}'],
                    validation_scores=training_metadata['training_history'][f'validation_{metric}'],
                    best_epoch=training_metadata[f'best_epoch_{metric}'] - 1,
                    metric=metric,
                    path=model_directory / f'learning_curve_fold_{fold}_{metric}.png'
                )
                settings.logger.info(f'Saved learning_curve_fold_{fold}_{metric}.png to {model_directory}')

            with open(model_directory / f'training_metadata.json', mode='w') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)
            settings.logger.info(f'Saved training_metadata.json to {model_directory}')

    elif args.mode == 'test':

        conditions = ['spinal_canal_stenosis', 'neural_foraminal_narrowing', 'subarticular_stenosis']
        levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        sides = ['left', 'right']
        label_mapping = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

        df_train_stack = []

        for _, row in tqdm(df_train.drop_duplicates(subset='study_id').iterrows(), total=df_train['study_id'].nunique()):
            for condition in conditions:
                    if condition != 'spinal_canal_stenosis':
                        for side in sides:
                            for level in levels:
                                target_column = f'{side}_{condition}_{level}'
                                df_train_stack.append({
                                    'study_id': row['study_id'],
                                    'condition': condition,
                                    'level': level,
                                    'side': side,
                                    'Normal/Mild': int(row[target_column] == 0) if pd.isnull(row[target_column]) is False else np.nan,
                                    'Moderate': int(row[target_column] == 1) if pd.isnull(row[target_column]) is False else np.nan,
                                    'Severe': int(row[target_column] == 2) if pd.isnull(row[target_column]) is False else np.nan
                                })
                    else:
                        for level in levels:
                            target_column = f'{condition}_{level}'
                            df_train_stack.append({
                                'study_id': row['study_id'],
                                'condition': condition,
                                'level': level,
                                'side': np.nan,
                                'Normal/Mild': int(row[target_column] == 0) if pd.isnull(row[target_column]) is False else np.nan,
                                'Moderate': int(row[target_column] == 1) if pd.isnull(row[target_column]) is False else np.nan,
                                'Severe': int(row[target_column] == 2) if pd.isnull(row[target_column]) is False else np.nan
                            })

        df_train_stack = pd.DataFrame(df_train_stack)
        df_train_stack = metrics.create_sample_weights(df_train_stack)

        # Set model, device and seed for reproducible results
        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        amp = config['training']['amp']

        test_folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']

        scores = []

        for fold, model_file_name in zip(test_folds, model_file_names):

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_directory / model_file_name))
            model.to(device)
            model.eval()

            validation_mask = df_train[fold] == 1
            settings.logger.info(
                f'''
                Fold {fold}
                Validation Size {np.sum(validation_mask)} ({np.sum(validation_mask) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            validation_dataset = torch_datasets.VolumeClassificationDataset(
                volume_paths=volume_paths[validation_mask],
                spinal_canal_stenosis_targets=targets['spinal_canal_stenosis'][validation_mask],
                neural_foraminal_narrowing_targets=targets['neural_foraminal_narrowing'][validation_mask],
                subarticular_stenosis_targets=targets['subarticular_stenosis'][validation_mask],
                spinal_canal_stenosis_sample_weights=sample_weights['spinal_canal_stenosis'][validation_mask],
                neural_foraminal_narrowing_sample_weights=sample_weights['neural_foraminal_narrowing'][validation_mask],
                subarticular_stenosis_sample_weights=sample_weights['subarticular_stenosis'][validation_mask],
                transforms=None
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            progress_bar = tqdm(validation_loader)

            validation_spinal_canal_stenosis_predictions = []
            validation_neural_foraminal_narrowing_predictions = []
            validation_subarticular_stenosis_predictions = []

            for step, (inputs, _, _) in enumerate(progress_bar):

                inputs = inputs.to(device)

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            spinal_canal_stenosis_outputs, neural_foraminal_narrowing_outputs, subarticular_stenosis_outputs = model(inputs)
                    else:
                        spinal_canal_stenosis_outputs, neural_foraminal_narrowing_outputs, subarticular_stenosis_outputs = model(inputs)

                validation_spinal_canal_stenosis_predictions.append(spinal_canal_stenosis_outputs.detach().cpu())
                validation_neural_foraminal_narrowing_predictions.append(neural_foraminal_narrowing_outputs.detach().cpu())
                validation_subarticular_stenosis_predictions.append(subarticular_stenosis_outputs.detach().cpu())

            validation_spinal_canal_stenosis_predictions = torch.softmax(torch.cat(validation_spinal_canal_stenosis_predictions, dim=0).float(), dim=1).numpy()
            validation_neural_foraminal_narrowing_predictions = torch.softmax(torch.cat(validation_neural_foraminal_narrowing_predictions, dim=0).float(), dim=1).numpy()
            validation_subarticular_stenosis_predictions = torch.softmax(torch.cat(validation_subarticular_stenosis_predictions, dim=0).float(), dim=1).numpy()

            for level_idx, level in enumerate(levels):
                validation_predictions = validation_spinal_canal_stenosis_predictions[:, :, level_idx]
                for label, class_idx in label_mapping.items():
                    df_train.loc[validation_mask, f'spinal_canal_stenosis_{level}_{label}_prediction'] = validation_predictions[:, class_idx]

            for side_idx, side in enumerate(sides):
                for level_idx, level in enumerate(levels):
                    validation_predictions = validation_neural_foraminal_narrowing_predictions[:, :, (side_idx * 5) + level_idx]
                    for label, class_idx in label_mapping.items():
                        df_train.loc[validation_mask, f'{side}_neural_foraminal_narrowing_{level}_{label}_prediction'] = validation_predictions[:, class_idx]

            for side_idx, side in enumerate(sides):
                for level_idx, level in enumerate(levels):
                    validation_predictions = validation_subarticular_stenosis_predictions[:, :, (side_idx * 5) + level_idx]
                    for label, class_idx in label_mapping.items():
                        df_train.loc[validation_mask, f'{side}_subarticular_stenosis_{level}_{label}_prediction'] = validation_predictions[:, class_idx]

            df_validation_study_predictions = df_train.loc[
                validation_mask,
                [column for column in df_train.columns if 'prediction' in column] + ['study_id']
            ].groupby('study_id').mean().reset_index()

            for idx, row in df_validation_study_predictions.iterrows():
                df_train_stack.loc[
                    df_train_stack['study_id'] == row['study_id'],
                    ['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']
                ] = row[1:].values.reshape(-1, 3)

            df_validation_spinal_canal_stenosis = df_train_stack.loc[
                (df_train_stack['study_id'].isin(df_validation_study_predictions['study_id'])) &
                (df_train_stack['condition'] == 'spinal_canal_stenosis')
            ]
            validation_spinal_canal_stenosis_scores = metrics.classification_scores(
                y_true=df_validation_spinal_canal_stenosis[['Normal/Mild', 'Moderate', 'Severe']].values,
                y_pred=df_validation_spinal_canal_stenosis[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
                sample_weights=df_validation_spinal_canal_stenosis['weight'].values,
            )

            df_validation_neural_foraminal_narrowing = df_train_stack.loc[
                (df_train_stack['study_id'].isin(df_validation_study_predictions['study_id'])) &
                (df_train_stack['condition'] == 'neural_foraminal_narrowing')
            ]
            validation_neural_foraminal_narrowing_scores = metrics.classification_scores(
                y_true=df_validation_neural_foraminal_narrowing[['Normal/Mild', 'Moderate', 'Severe']].values,
                y_pred=df_validation_neural_foraminal_narrowing[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
                sample_weights=df_validation_neural_foraminal_narrowing['weight'].values,
            )

            df_validation_subarticular_stenosis = df_train_stack.loc[
                (df_train_stack['study_id'].isin(df_validation_study_predictions['study_id'])) &
                (df_train_stack['condition'] == 'subarticular_stenosis')
            ]
            validation_subarticular_stenosis_scores = metrics.classification_scores(
                y_true=df_validation_subarticular_stenosis[['Normal/Mild', 'Moderate', 'Severe']].values,
                y_pred=df_validation_subarticular_stenosis[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
                sample_weights=df_validation_subarticular_stenosis['weight'].values,
            )

            df_validation_any_spinal = df_train_stack.loc[
                (df_train_stack['study_id'].isin(df_validation_study_predictions['study_id'])) &
                (df_train_stack['condition'] == 'spinal_canal_stenosis')
            ].groupby('study_id')[['Severe', 'Severe_prediction', 'weight']].max().reset_index()
            validation_any_spinal_scores = metrics.classification_scores(
                y_true=df_validation_any_spinal['Severe'].values,
                y_pred=df_validation_any_spinal['Severe_prediction'].values,
                sample_weights=df_validation_any_spinal['weight'].values,
            )

            fold_scores = {}
            validation_condition_scores = [
                validation_spinal_canal_stenosis_scores, validation_neural_foraminal_narrowing_scores,
                validation_subarticular_stenosis_scores, validation_any_spinal_scores
            ]
            for condition, condition_scores in zip(conditions + ['any_spinal'], validation_condition_scores):
                for metric, score in condition_scores.items():
                    fold_scores[f'{condition}_{metric}'] = score

            validation_average_scores = pd.DataFrame(validation_condition_scores).mean().to_dict()
            fold_scores['average_log_loss'] = validation_average_scores['log_loss']
            fold_scores['average_sample_weighted_log_loss'] = validation_average_scores['sample_weighted_log_loss']

            settings.logger.info(f'{fold} Validation Scores\n{json.dumps(fold_scores, indent=2)}')
            scores.append(fold_scores)

        scores = pd.DataFrame(scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        df_oof_spinal_canal_stenosis = df_train_stack.loc[df_train_stack['condition'] == 'spinal_canal_stenosis']
        oof_spinal_canal_stenosis_scores = metrics.classification_scores(
            y_true=df_oof_spinal_canal_stenosis[['Normal/Mild', 'Moderate', 'Severe']].values,
            y_pred=df_oof_spinal_canal_stenosis[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
            sample_weights=df_oof_spinal_canal_stenosis['weight'].values,
        )

        df_oof_neural_foraminal_narrowing = df_train_stack.loc[df_train_stack['condition'] == 'neural_foraminal_narrowing']
        oof_neural_foraminal_narrowing_scores = metrics.classification_scores(
            y_true=df_oof_neural_foraminal_narrowing[['Normal/Mild', 'Moderate', 'Severe']].values,
            y_pred=df_oof_neural_foraminal_narrowing[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
            sample_weights=df_oof_neural_foraminal_narrowing['weight'].values,
        )

        df_oof_subarticular_stenosis = df_train_stack.loc[df_train_stack['condition'] == 'subarticular_stenosis']
        oof_subarticular_stenosis_scores = metrics.classification_scores(
            y_true=df_oof_subarticular_stenosis[['Normal/Mild', 'Moderate', 'Severe']].values,
            y_pred=df_oof_subarticular_stenosis[['Normal/Mild_prediction', 'Moderate_prediction', 'Severe_prediction']].values,
            sample_weights=df_oof_subarticular_stenosis['weight'].values,
        )

        df_oof_any_spinal = df_train_stack.loc[
            df_train_stack['condition'] == 'spinal_canal_stenosis'
        ].groupby('study_id')[['Severe', 'Severe_prediction', 'weight']].max().reset_index()
        oof_any_spinal_scores = metrics.classification_scores(
            y_true=df_oof_any_spinal['Severe'].values,
            y_pred=df_oof_any_spinal['Severe_prediction'].values,
            sample_weights=df_oof_any_spinal['weight'].values,
        )

        oof_scores = {}
        oof_condition_scores = [
            oof_spinal_canal_stenosis_scores, oof_neural_foraminal_narrowing_scores,
            oof_subarticular_stenosis_scores, oof_any_spinal_scores
        ]
        for condition, condition_scores in zip(conditions + ['any_spinal'], oof_condition_scores):
            for metric, score in condition_scores.items():
                oof_scores[f'{condition}_{metric}'] = score

        oof_average_scores = pd.DataFrame(oof_condition_scores).mean().to_dict()
        oof_scores['average_log_loss'] = oof_average_scores['log_loss']
        oof_scores['average_sample_weighted_log_loss'] = oof_average_scores['sample_weighted_log_loss']

        settings.logger.info(f'OOF Scores\n{json.dumps(oof_scores, indent=2)}')

        scores = pd.concat((
            scores,
            pd.DataFrame([oof_scores])
        )).reset_index(drop=True)
        scores['fold'] = test_folds + ['OOF']
        scores = scores[scores.columns.tolist()[::-1]]
        scores.to_csv(model_directory / 'scores.csv', index=False)
        settings.logger.info(f'scores.csv is saved to {model_directory}')

        visualization.visualize_scores(
            scores=scores,
            title=f'Fold and OOF Scores of {len(test_folds)} Model(s)',
            path=model_directory / 'scores.png'
        )
        settings.logger.info(f'scores.png is saved to {model_directory}')

        df_train_stack.to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')