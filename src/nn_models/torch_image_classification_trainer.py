import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
import torch.optim as optim

sys.path.append('..')
import settings
import torch_datasets
import torch_modules
import torch_utilities
import transforms
import metrics
import visualization


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

    validation_predictions: torch.Tensor of shape (n_samples, n_outputs)
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

    image_paths, _, targets = torch_datasets.prepare_dataset(df=df, features=None)
    image_transforms = transforms.get_image_transforms(**config['transforms'])
    training_datasets = config['dataset']['training']
    validation_datasets = config['dataset']['validation']
    negative_sample_count = config['dataset']['negative_sample_count']

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        training_metadata = {}
        training_folds = config['training']['folds']

        for fold in training_folds:

            training_mask = (df[f'fold{fold}'] == 0) & (df['dataset'].isin(training_datasets))
            validation_mask = (df[f'fold{fold}'] == 1) & (df['dataset'].isin(validation_datasets))

            np.random.seed(42)
            training_positive_idx = np.where(training_mask & (df['target'] == 1))[0]
            training_negative_idx = []
            for dataset, count in negative_sample_count.items():
                training_negative_idx_dataset = np.random.choice(np.where(training_mask & (df['target'] == 0) & (df['dataset'] == dataset))[0], count)
                training_negative_idx.append(training_negative_idx_dataset)
            training_negative_idx = np.concatenate(training_negative_idx)
            training_idx = np.concatenate((training_positive_idx, training_negative_idx))

            settings.logger.info(
                f'''
                Fold {fold}
                Training Size {training_idx.shape[0]} ({training_idx.shape[0] // config["training"]["training_batch_size"] + 1} steps)
                Validation Size: {np.sum(validation_mask)} ({np.sum(validation_mask) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            weights = np.ones_like(training_idx).astype(int)
            positive_weight = (df.loc[training_idx, 'target'] == 0).sum() / (df.loc[training_idx, 'target'] == 1).sum()
            weights[np.where(df.loc[training_idx, 'target'] == 1)[0]] = positive_weight
            training_sampler = ExhaustiveWeightedRandomSampler(weights.tolist(), num_samples=weights.shape[0])

            training_dataset = torch_datasets.TabularImageDataset(
                image_paths=image_paths[training_idx],
                features=None,
                targets=targets[training_idx],
                transforms=image_transforms['training']
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=training_sampler,
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

            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model_checkpoint_path = config['model']['model_checkpoint_path']
            if model_checkpoint_path is not None:
                model_checkpoint_path = settings.MODELS / model_checkpoint_path
                model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
            model.to(device)

            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            epochs = config['training']['epochs']
            best_epoch = 1
            training_history = {f'{dataset}_{metric}': [] for metric in config['persistence']['save_best_metrics'] for dataset in ['training', 'validation']}

            for epoch in range(1, epochs + 1):

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
                    model_name = f'model_fold_{fold}_epoch_{epoch}.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory}')

                for metric, higher_or_lower in zip(config['persistence']['save_best_metrics'], config['persistence']['save_best_metric_higher_or_lower']):

                    last_validation_metric = validation_results[metric]

                    if higher_or_lower == 'lower':
                        best_validation_metric = np.min(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else np.inf
                        save_condition = last_validation_metric < best_validation_metric
                    elif higher_or_lower == 'higher':
                        best_validation_metric = np.max(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else -np.inf
                        save_condition = last_validation_metric > best_validation_metric
                    else:
                        raise ValueError(f'Invalid save_best_metric_higher_or_lower value {higher_or_lower}')

                    if save_condition:

                        previous_model = glob(str(model_directory / f'model_fold_{fold}_epoch_*_best_{metric}*'))
                        if len(previous_model) > 0:
                            os.remove(previous_model[0])
                            settings.logger.info(f'Deleted {previous_model[0].split("/")[-1]} from {model_directory}')

                        model_name = f'model_fold_{fold}_epoch_{epoch}_best_{metric}_{last_validation_metric:.6f}.pt'
                        torch.save(model.state_dict(), model_directory / model_name)
                        settings.logger.info(f'Saved {model_name} to {model_directory} (validation {metric} increased/decreased from {best_validation_metric:.6f} to {last_validation_metric:.6f})\n')

                    if metric in training_results.keys():
                        training_history[f'training_{metric}'].append(training_results[metric])
                    else:
                        training_history[f'training_{metric}'].append(np.nan)
                    training_history[f'validation_{metric}'].append(validation_results[metric])

                training_metadata['training_history'] = training_history

            for metric, higher_or_lower in zip(config['persistence']['save_best_metrics'], config['persistence']['save_best_metric_higher_or_lower']):

                if higher_or_lower == 'lower':
                    best_epoch = int(np.argmin(training_history[f'validation_{metric}']))
                elif higher_or_lower == 'higher':
                    best_epoch = int(np.argmax(training_history[f'validation_{metric}']))
                else:
                    raise ValueError(f'Invalid save_best_metric_higher_or_lower value {higher_or_lower}')

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

        # Set model, device and seed for reproducible results
        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        amp = config['training']['amp']

        test_folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']
        tta_indices = config['test']['tta_indices']

        scores = []
        curves = {dataset: [] for dataset in validation_datasets + ['global']}

        df['prediction'] = np.nan

        for fold, model_file_name in zip(test_folds, model_file_names):

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_directory / model_file_name, weights_only=True))
            model.to(device)
            model.eval()

            validation_mask = (df[f'fold{fold}'] == 1) & (df['dataset'].isin(validation_datasets))
            settings.logger.info(
                f'''
                Fold {fold}
                Validation Size {np.sum(validation_mask)} ({np.sum(validation_mask) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            validation_dataset = torch_datasets.TabularImageDataset(
                image_paths=image_paths[validation_mask],
                features=None,
                targets=None,
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

            validation_predictions = []
            validation_features = []

            for inputs in tqdm(validation_loader):

                inputs = inputs.to(device)
                batch_outputs = torch.zeros((inputs.shape[0], 1)).to(device)
                batch_features = torch.zeros((inputs.shape[0], 1280)).to(device)

                for tta_idx in tta_indices:

                    with torch.no_grad():
                        if amp:
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                outputs, features = model(transforms.get_tta_inputs(inputs, tta_idx=tta_idx), extract_features=True)

                        else:
                            outputs, features = model(transforms.get_tta_inputs(inputs, tta_idx=tta_idx), extract_features=True)

                    batch_outputs += outputs / len(tta_indices)
                    batch_features += features / len(tta_indices)

                validation_predictions.append(batch_outputs.cpu())
                validation_features.append(batch_features.cpu())

            validation_predictions = torch.sigmoid(torch.cat(validation_predictions).float()).numpy()
            df.loc[validation_mask, 'prediction'] = validation_predictions
            df.loc[validation_mask, 'prediction_rank'] = df.loc[validation_mask, 'prediction'].rank(pct=True)

            validation_features = torch.cat(validation_features).float().numpy()
            df.loc[validation_mask, [f'image_feature_{i}' for i in range(validation_features.shape[1])]] = validation_features

            validation_scores = {}
            if len(validation_datasets) > 1:
                for dataset in validation_datasets:
                    dataset_validation_mask = validation_mask & (df['dataset'] == dataset)
                    dataset_validation_scores = metrics.classification_scores(
                        y_true=df.loc[dataset_validation_mask, 'target'],
                        y_pred=df.loc[dataset_validation_mask, 'prediction_rank'],
                    )
                    dataset_validation_scores = {f'{dataset}_{k}': v for k, v in dataset_validation_scores.items()}
                    validation_scores.update(dataset_validation_scores)

                    dataset_validation_curves = metrics.classification_curves(
                        y_true=df.loc[dataset_validation_mask, 'target'],
                        y_pred=df.loc[dataset_validation_mask, 'prediction_rank'],
                    )
                    curves[dataset].append(dataset_validation_curves)

            global_validation_scores = metrics.classification_scores(
                y_true=df.loc[validation_mask, 'target'],
                y_pred=df.loc[validation_mask, 'prediction_rank'],
            )
            validation_scores.update(global_validation_scores)
            settings.logger.info(f'{fold} Validation Scores\n{json.dumps(validation_scores, indent=2)}')
            scores.append(validation_scores)

            global_validation_curves = metrics.classification_curves(
                y_true=df.loc[validation_mask, 'target'],
                y_pred=df.loc[validation_mask, 'prediction_rank'],
            )
            curves['global'].append(global_validation_curves)

        scores = pd.DataFrame(scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        oof_scores = {}
        if len(validation_datasets) > 1:
            for dataset in validation_datasets:
                dataset_mask = df['dataset'] == dataset
                dataset_oof_scores = metrics.classification_scores(
                    y_true=df.loc[dataset_mask, 'target'],
                    y_pred=df.loc[dataset_mask, 'prediction_rank'],
                )
                dataset_oof_scores = {f'{dataset}_{k}': v for k, v in dataset_oof_scores.items()}
                oof_scores.update(dataset_oof_scores)

        oof_mask = df['prediction_rank'].notna()
        global_oof_scores = metrics.classification_scores(
            y_true=df.loc[oof_mask, 'target'],
            y_pred=df.loc[oof_mask, 'prediction_rank'],
        )
        oof_scores.update(global_oof_scores)
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

        if len(validation_datasets) > 1:

            for dataset in validation_datasets:
                dataset_mask = df['dataset'] == dataset

                visualization.visualize_roc_curves(
                    roc_curves=[curve['roc'] for curve in curves[dataset]],
                    title=f'{dataset} Validation ROC Curves',
                    path=model_directory / f'roc_curves_{dataset}.png'
                )
                settings.logger.info(f'roc_curves_{dataset}.png is saved to {model_directory}')

                visualization.visualize_pr_curves(
                    pr_curves=[curve['pr'] for curve in curves[dataset]],
                    title=f'{dataset} Validation PR Curves',
                    path=model_directory / f'pr_curves_{dataset}.png'
                )
                settings.logger.info(f'pr_curves_{dataset}.png is saved to {model_directory}')

                visualization.visualize_predictions(
                    y_true=df.loc[dataset_mask, 'target'],
                    y_pred=df.loc[dataset_mask, 'prediction_rank'],
                    title=f'{dataset} Predictions Histogram',
                    path=model_directory / f'predictions_{dataset}.png'
                )
                settings.logger.info(f'predictions_{dataset}.png is saved to {model_directory}')

        visualization.visualize_roc_curves(
            roc_curves=[curve['roc'] for curve in curves['global']],
            title='Global Validation ROC Curves',
            path=model_directory / 'roc_curves_global.png'
        )
        settings.logger.info(f'roc_curves_global.png is saved to {model_directory}')

        visualization.visualize_pr_curves(
            pr_curves=[curve['pr'] for curve in curves['global']],
            title='Global Validation PR Curves',
            path=model_directory / 'pr_curves_global.png'
        )
        settings.logger.info(f'pr_curves_global.png is saved to {model_directory}')

        visualization.visualize_predictions(
            y_true=df.loc[oof_mask, 'target'],
            y_pred=df.loc[oof_mask, 'prediction_rank'],
            title='Global Predictions Histogram',
            path=model_directory / 'predictions_global.png'
        )
        settings.logger.info(f'predictions_global.png is saved to {model_directory}')

        df.loc[oof_mask, ['isic_id', 'prediction', 'prediction_rank']].to_parquet(model_directory / 'oof_predictions.parquet')
        settings.logger.info(f'oof_predictions.parquet is saved to {model_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')
