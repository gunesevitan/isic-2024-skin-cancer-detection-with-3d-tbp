## 1. Overview

* GitHub Repository: https://github.com/gunesevitan/isic-2024-skin-cancer-detection-with-3d-tbp
* Notebook: https://www.kaggle.com/code/gunesevitan/isic-2024-inference
* Dataset: https://www.kaggle.com/datasets/gunesevitan/isic-2024-dataset

My solution can be divided into 3 sections:

* Image Features
* Metadata Features
* Patient Context Features

## 2. Image Features

I decided to use two models for images.

* A model that can generate robust image embeddings for computing image similarities (DINOv2 Small)
* A shallow model for generating relative melanoma scores (EfficientNet B0)

### 2.1. Embedding Model

I used DINOv2 small model for extracting image embeddings. I didn't fine-tune it. After getting model outputs for each image, I max pooled the last hidden states except cls token and normalized them. This gave me a `(n_images, 384)` shaped matrix.

```
dino_outputs = dino_outputs.last_hidden_state[:, 1:].max(dim=1)[0]
dino_outputs = F.normalize(dino_outputs, dim=-1, p=2).cpu()
```

### 2.2. Melanoma Score Model

I thought a shallow model would be better since the images were too small and receptive field could be an issue. I only used `efficientnet_b0` from `timm` library with ImageNet pretrained weights.

I used a custom weighted sampler that ensures sampling all data points in each epoch and getting at least one positive sample on each batch.

I used the following augmentations in my training pipeline.

* AdaptiveResize (use different algorithms for upsampling/downsampling)

```
image_height, image_width = inputs.shape[:2]

if self.resize_height > image_height or self.resize_width > image_width:
    if len(self.upsample_interpolations) > 1:
        interpolation = np.random.choice(self.upsample_interpolations)
    else:
        interpolation = self.upsample_interpolations[0]
else:
    if len(self.downsample_interpolations) > 1:
        interpolation = np.random.choice(self.downsample_interpolations)
    else:
        interpolation = self.downsample_interpolations[0]

inputs = cv2.resize(inputs, dsize=(self.resize_width, self.resize_height), interpolation=interpolation)
```

* Transpose
* Vertical Flip
* Horizontal Flip
* Random Hue/Saturation/Value
* ImageNet Normalization

Those are the configurations of the listed augmentations.

```
transforms:
  resize_height: 224
  resize_width: 224
  transpose_probability: 0.5
  vertical_flip_probability: 0.5
  horizontal_flip_probability: 0.5
  brightness_limit: 0.2
  contrast_limit: 0.2
  random_brightness_contrast_probability: 0.75
  hue_shift_limit: 20
  sat_shift_limit: 20
  val_shift_limit: 20
  hue_saturation_value_probability: 0.5
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

I used 4 test-time augmentations (TTA) along with raw images while getting predictions. Raw and TTA predictions are rank transformed and merged with average pooling. Applied TTAs are:

* Transpose
* Vertical Flip
* Horizontal Flip
* Diagonal Flip

Those are the final scores of this model. I didn't submit it, so I don't know its LB score.

| fold    | partial_auc            | roc_auc                |
|---------|------------------------|------------------------|
| 1       | 0.16766166797498847    | 0.9461433005229546     |
| 2       | 0.15783020634385644    | 0.9383655945245191     |
| 3       | 0.16189234413265044    | 0.9404248217077322     |
| 4       | 0.16237978380239065    | 0.9448725463868151     |
| 5       | 0.1505018265474405     | 0.922693663556354      |
| **OOF** | **0.1599224700542807** | **0.9389163766247772** |


OOF rank transformed predictions are merged to training set and fold average is taken on test time.  

## 3. Metadata Features

This is the section where I created column-wise interactions of metadata features.

* LAB space lesion x lesion ratios and differences
* LAB space external x external ratios and differences
* LAB space ratios of lesion x lesion and external x external differences
* LAB space lesion x external ratios and differences
* LAB space norm ratios and differences
* Diameter ratios, differences and mean
* Diameter x perimeter ratios and differences
* Area x diameter squared ratios and differences
* Area x perimeter squared ratios and differences
* Circularity and shape index
* Distance to origin
* XY, XZ, YZ angle
* Border irregularity column-wise aggregations
* Color irregularity column-wise aggregations
* All irregularity column-wise aggregations

I tried converting LAB space to HSV and RGB, but it didn't work.

## 4. Patient Context Features

This is the most important section and crucial for this task. I saw this passage on competition overview and my whole feature engineering intuition is built around it.

> Benign moles on an individual tend to resemble each other in terms of color, shape, size, and pattern. Outlier lesions are more likely to be melanoma, an observation known as the “ugly duckling sign”. However, most skin lesion classification algorithms are trained for independent analysis of individual skin lesions. The dataset presented here is novel because it represents each person's lesion phenotype more completely. Algorithms may be able to enhance their diagnostic accuracy when taking into account “context” within the same patient to determine which images represent a cancerous outlier.

### 4.1. Aggregations

I did very few aggregations on patient level because they didn't work well for me, but some of them were essential like age property. Extracting visits from patient ages gave me a significant boost (0.002x).

```
df['patient_visit_id'] = df['age_approx'] - df.groupby('patient_id')['age_approx'].transform('min')
df['patient_visit_count'] = df.groupby('patient_id')['patient_visit_id'].transform('nunique')
df['patient_visit_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id'])['patient_id'].transform('count')
df['patient_visit_site_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'anatom_site_general'])['patient_id'].transform('count')
df['patient_visit_location_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'tbp_lv_location'])['patient_id'].transform('count')
df['patient_visit_location_simple_lesion_count'] = df.groupby(['patient_id', 'patient_visit_id', 'tbp_lv_location_simple'])['patient_id'].transform('count')
```

Rest of this subsection is basically taking mean, median, std, min, max, skew of all features on patient, patient/site, patient/location, patient/location_simple groups. Most of them didn't work well, but some of them were used in my final feature set.

### 4.2. Normalized Ranks

This is where I got the largest score boost since it provides position of each lesion w.r.t. other lesions which is ugly duckling in a nutshell.
 
Raw features, metadata features and image model predictions are ranked inside patient/site, patient/location, patient/location_simple groups.
```
rank_columns = [
    'tbp_lv_nevi_confidence',
    'tbp_lv_A', 'tbp_lv_Aext',
    'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext',
    
    ...
    
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
```

### 4.3. Anchor Features

I called this feature group anchor features because I was selecting an anchor lesion based on a given feature and compare each lesion with the selected anchor lesion using ratios and differences. It is inspired from triplet loss.

I only used `clin_size_long_diam_mm`, `image_model_prediction_rank` as anchor features and raw LAB features for ratios and differences. It is easier to explain it on code.

```
anchor_columns = ['clin_size_long_diam_mm', 'image_model_prediction_rank']
feature_columns = [
    'tbp_lv_A', 'tbp_lv_Aext',
    'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext',
    
    ...

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
```

### 4.4. Cluster Features

I fit DBSCAN on patient/site, patient/location, patient/location_simple groups using different feature groups such as

```
color_columns = [
    'tbp_lv_A', 'tbp_lv_Aext',
    'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_H', 'tbp_lv_Hext'
]
lesion_color_columns = [
    'tbp_lv_A',
    'tbp_lv_B',
    'tbp_lv_C',
    'tbp_lv_L',
    'tbp_lv_H'
]
outside_color_columns = [
    'tbp_lv_Aext',
    'tbp_lv_Bext',
    'tbp_lv_Cext',
    'tbp_lv_Lext',
    'tbp_lv_Hext'
]
shape_columns = [
    'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
]
location_columns = ['tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
```

For every group with more than two lesions, I minmax normalized the feature group and assigned cluster labels to corresponding rows.

```
for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

    if df_group.shape[0] > 2:

        idx = (df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site)

        for columns, column_group_name in zip(column_groups, column_group_names):

            min_max_scaler = MinMaxScaler()
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            dbscan.fit(min_max_scaler.fit_transform(df_group[columns]))
            df.loc[idx, f'patient_site_{column_group_name}_cluster'] = dbscan.labels_
```

Then I counted lesions in each cluster, and calculated ratios and differences of cluster lesion counts to group lesion counts. 

```
df['patient_site_color_cluster_count'] = df.groupby(['patient_id', 'anatom_site_general_filled', 'patient_site_color_cluster'])['patient_site_color_cluster'].transform('count')
df['patient_site_color_cluster_ratio'] = df['patient_site_color_cluster_count'] / df['patient_site_lesion_count']
df['patient_site_color_cluster_difference'] = df['patient_site_color_cluster_count'] - df['patient_site_lesion_count']
```

### 4.5. Metadata Distance Features

I calculated pairwise euclidean distances of each lesion inside patient/site, patient/location, patient/location_simple groups
using the same feature groups listed in previous section.

I calculated mean, median, std, min, max distances among those groups which were very beneficial for my models. I also calculated ratios and differences between each lesion's most and least similar counterpart which also mimics ugly duckling.

```
for (patient_id,), df_group in tqdm(df.groupby(['patient_id']), total=1042):

    if df_group.shape[0] > 2:

        idx = df['patient_id'] == patient_id

        for columns, column_group_name in zip(column_groups, column_group_names):

            min_max_scaler = MinMaxScaler()
            pairwise_distances = squareform(pdist(min_max_scaler.fit_transform(df_group[columns])))
            np.fill_diagonal(pairwise_distances, np.nan)

            mean_distances = np.nanmean(pairwise_distances, axis=1)
            median_distances = np.nanmedian(pairwise_distances, axis=1)
            std_distances = np.nanstd(pairwise_distances, axis=1)
            min_distances = np.nanmin(pairwise_distances, axis=1)
            max_distances = np.nanmax(pairwise_distances, axis=1)
            min_distance_idx = np.nanargmin(pairwise_distances, axis=1)
            max_distance_idx = np.nanargmax(pairwise_distances, axis=1)

            df.loc[idx, f'patient_{column_group_name}_mean_distance'] = mean_distances
            df.loc[idx, f'patient_{column_group_name}_median_distance'] = median_distances
            df.loc[idx, f'patient_{column_group_name}_std_distance'] = std_distances
            df.loc[idx, f'patient_{column_group_name}_min_distance'] = min_distances
            df.loc[idx, f'patient_{column_group_name}_max_distance'] = max_distances
            df.loc[idx, f'patient_{column_group_name}_min_distance_idx'] = min_distance_idx
            df.loc[idx, f'patient_{column_group_name}_max_distance_idx'] = max_distance_idx

            df_patient = df.loc[idx].reset_index(drop=True)
            df_patient_min_aligned = df_patient.iloc[df_patient[f'patient_{column_group_name}_min_distance_idx'].values].reset_index(drop=True)
            df_patient_max_aligned = df_patient.iloc[df_patient[f'patient_{column_group_name}_max_distance_idx'].values].reset_index(drop=True)

            df_patient_min_differences = df_patient[difference_and_ratio_columns] - df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_differences = df_patient_min_differences.rename(columns={column: f'patient_{column}_min_{column_group_name}_distance_lesion_difference' for column in df_patient_min_differences.columns})
            df.loc[idx, df_patient_min_differences.columns.tolist()] = df_patient_min_differences.values

            df_patient_min_ratios = df_patient[difference_and_ratio_columns] / df_patient_min_aligned[difference_and_ratio_columns]
            df_patient_min_ratios = df_patient_min_ratios.rename(columns={column: f'patient_{column}_min_{column_group_name}_distance_lesion_ratio' for column in df_patient_min_ratios.columns})
            df.loc[idx, df_patient_min_ratios.columns.tolist()] = df_patient_min_ratios.values

            df_patient_max_differences = df_patient[difference_and_ratio_columns] - df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_differences = df_patient_max_differences.rename(columns={column: f'patient_{column}_max_{column_group_name}_distance_lesion_difference' for column in df_patient_max_differences.columns})
            df.loc[idx, df_patient_max_differences.columns.tolist()] = df_patient_max_differences.values

            df_patient_max_ratios = df_patient[difference_and_ratio_columns] / df_patient_max_aligned[difference_and_ratio_columns]
            df_patient_max_ratios = df_patient_max_ratios.rename(columns={column: f'patient_{column}_max_{column_group_name}_distance_lesion_ratio' for column in df_patient_max_ratios.columns})
            df.loc[idx, df_patient_max_ratios.columns.tolist()] = df_patient_max_ratios.values
```

### 4.6. Image Distance Features

I created the same distance features for image embeddings as well on exact same columns. The only difference is I took dot product of image embeddings with itself.

```
for (patient_id, site), df_group in tqdm(df.groupby(['patient_id', 'anatom_site_general_filled']), total=5042):

    if df_group.shape[0] > 2:

        idx = np.where((df['patient_id'] == patient_id) & (df['anatom_site_general_filled'] == site))[0]

        patient_image_embeddings = image_embeddings[idx]
        pairwise_distances = (patient_image_embeddings @ patient_image_embeddings.T).cpu().numpy()
        np.fill_diagonal(pairwise_distances, np.nan)
        
        ...
```

## 5. Validation

I used stratified group kfold for validation. `patient_id` used as the group column and a custom stratify column is created by merging `iddx_1` and `iddx_3`.

```
df_train_metadata['stratify_column'] = df_train_metadata['iddx_1'].values
df_train_metadata.loc[df_train_metadata['target'] == 1, 'stratify_column'] = df_train_metadata.loc[df_train_metadata['target'] == 1, 'iddx_3'].values
```

I run this split for 100 seeds and take the seed that generates folds with most similar target means, validation sizes and patient lesion count means.

## 6. Models

I only used LightGBM models for binary and multiclass classification. Besides the task type, I also had two feature sets; only tabular features and tabular + image features, so in total I used 4 LightGBM models. Each of those models are trained for 5 seeds and 5 folds. In test time, there were 25 (`5 fold * 5 seed`) predictions generated from a single model.

For binary classification, each fold and seed predictions are rank transformed.  For multiclass classification, `iddx_1` is used as the target and softmaxed melanoma probability is rank transformed for each fold and seed. I tried ranking task type using patient groups, but scores weren't competitive.

After executing the feature engineering steps, I had approximately 6000 features. I used manual feature selection by iterating one feature group at a time. For example, I select features from patient ranks, freeze that subset, and then move on to another group. I also tuned my hyperparameters manually cuz I don't like automated tuning. Feature selection and hyperparameter tuning is done using 5 seeds of binary classification model because multiclass classification model was too slow. I also didn't compute partial AUC during the training because it was very slow.

### 6.1. Hyperparameters:

**Binary:**

```
model_parameters:
  num_leaves: 128
  learning_rate: 0.01
  bagging_fraction: 0.7
  bagging_freq: 1
  feature_fraction: 0.4
  feature_fraction_bynode: 0.8
  min_data_in_leaf: 100
  min_gain_to_split: 0.
  lambda_l1: 0.
  lambda_l2: 0.
  max_bin: 255
  min_data_in_bin: 3
  max_depth: -1
  min_data_per_group: 100
  max_cat_threshold: 32
  cat_l2: 10
  cat_smooth: 10
  max_cat_to_onehot: 32
  boost_from_average: True
  objective: 'binary'
  scale_pos_weight: 1.
  metric: 'binary_logloss'
  seed: null
  feature_fraction_seed: null
  bagging_seed: null
  drop_seed: null
  data_random_seed: null
  boosting_type: 'gbdt'
  verbose: 1
  n_jobs: 16

fit_parameters:
  boosting_rounds: 700
  log_evaluation: 100
```

**Multiclass:**

```
model_parameters:
  num_leaves: 128
  learning_rate: 0.01
  bagging_fraction: 0.7
  bagging_freq: 1
  feature_fraction: 0.4
  feature_fraction_bynode: 0.8
  min_data_in_leaf: 100
  min_gain_to_split: 0.
  lambda_l1: 0.
  lambda_l2: 0.
  max_bin: 255
  min_data_in_bin: 3
  max_depth: -1
  min_data_per_group: 100
  max_cat_threshold: 32
  cat_l2: 10
  cat_smooth: 10
  max_cat_to_onehot: 32
  boost_from_average: True
  objective: 'multiclass'
  num_class: 3
  metric: 'multi_logloss'
  seed: null
  feature_fraction_seed: null
  bagging_seed: null
  drop_seed: null
  data_random_seed: null
  boosting_type: 'gbdt'
  verbose: 1
  n_jobs: 16

fit_parameters:
  boosting_rounds: 600
  log_evaluation: 100
```

### 6.2. LightGBM Tabular Binary (314 features)


| fold    | partial_auc             | roc_auc                |
|---------|-------------------------|------------------------|
| 1       | 0.1896690716020659      | 0.9878075189002438     |
| 2       | 0.1821128858988451      | 0.9757199198445059     |
| 3       | 0.18498914520934626     | 0.98140004243403       |
| 4       | 0.18381866705316866     | 0.9803265318526644     |
| 5       | 0.171643740582275       | 0.9644720994493647     |
| **OOF** | **0.18240074995234712** | **0.9783029893903495** |

### 6.3. LightGBM Tabular Multiclass (314 features)

| fold    | partial_auc             | roc_auc                |
|---------|-------------------------|------------------------|
| 1       | 0.18938518006975802     | 0.9875198311439575     |
| 2       | 0.18280268134666508     | 0.976432698683338      |
| 3       | 0.18496955414140664     | 0.9814527263067234     |
| 4       | 0.18358144755324654     | 0.9798427712193544     |
| 5       | 0.17264394575535286     | 0.9662676807854598     |
| **OOF** | **0.18266753222297086** | **0.9786197234784746** |

### 6.4. LightGBM Tabular + Image Binary

| fold    | partial_auc             | roc_auc                |
|---------|-------------------------|------------------------|
| 1       | 0.189646838933809       | 0.9879042400850883     |
| 2       | 0.18581242175183368     | 0.9802417000953847     |
| 3       | 0.18563367380983462     | 0.981664497940228      |
| 4       | 0.18920174966183503     | 0.9861041913161532     |
| 5       | 0.18084402735375438     | 0.9753169362679542     |
| **OOF** | **0.18618073871380736** | **0.9824628317007399** |

### 6.5. LightGBM Tabular + Image Multiclass

| fold    | partial_auc             | roc_auc              |
|---------|-------------------------|----------------------|
| 1       | 0.19011694350492486     | 0.9883906519139901   |
| 2       | 0.1859788391258087      | 0.9807986618232658   |
| 3       | 0.18534014254454648     | 0.9814523277902885   |
| 4       | 0.18934182254590465     | 0.9862962491091118   |
| 5       | 0.18256910738121462     | 0.9774681750841653   |
| **OOF** | **0.18668520983808776** | **0.98306873445027** |

## 7. Conclusion

I didn't get much benefit from blending, but I selected two blended submissions regardless of that. I think I was getting 0.0002 boost from blending or something like that. The selected submissions are:

* `LightGBM Tabular Binary * 0.15 + LightGBM Tabular + Image Binary * 0.85`
* `LightGBM Tabular Binary * 0.075 + LightGBM Tabular Multiclass * 0.075 + LightGBM Tabular + Image Binary * 0.425 +  LightGBM Tabular + Image Multiclass * 0.425`

First one scored 0.18 on public and 0.169 on private. Second scored 0.179 on public and 0.169 on private. Second one was my best private score submission, and I was able to select it.

I think the reason why I placed so low on the leaderboard is the distribution shift. We can't do much about it other than embracing the fall. It was a good feature engineering practice for me.
