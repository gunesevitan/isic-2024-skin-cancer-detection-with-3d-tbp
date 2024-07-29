import sys
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    metadata_directory = settings.DATA / 'datasets'

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    isic_datasets = [
        'ham10000-metadata', 'prove-ai-metadata', 'bcn20000-metadata',
        'isbi-2016-metadata', 'consecutive-biopsies-2020-metadata',
        'consumer-ai-apps-metadata', 'easy-dermoscopy-expert-agreement-study-metadata',
        'hiba-skin-lesions-metadata', 'longitudinal-images-with-various-types-metadata',
        'melanoma-and-nevus-dermoscopy-images-metadata', 'msk-1-metadata', 'msk-2-metadata',
        'msk-3-metadata', 'msk-4-metadata', 'msk-5-metadata', 'newly-acquired-melanoma-metadata',
        'repeated-dermoscopic-images-metadata', 'sonic-metadata', 'uda-1-metadata', 'uda-2-metadata',
        'isic-2016-metadata', 'isic-2017-metadata', 'isic-2018-metadata',
        'isic-2019-metadata', 'isic-2020-metadata', 'isic-2024-metadata',
    ]
    columns = [
        'id', 'isic_id', 'target', 'dataset', 'height', 'width', 'image_path'
    ]
    df_metadata = []
    for dataset in tqdm(isic_datasets):
        df = pd.read_parquet(metadata_directory / f'{dataset}.parquet')
        if dataset == 'isic-2024-metadata':
            df['id'] = pd.Series(df.index).apply(lambda x: f'isic_2024_{x}')
            df['dataset'] = 'isic_2024'
        df_metadata.append(df[columns])
    df_metadata = pd.concat(df_metadata).reset_index(drop=True)
    settings.logger.info(
        f'''
        ISIC Datasets Raw Target Counts
        {df_metadata.groupby('dataset')['target'].value_counts()}
        '''
    )

    df_isic_2020_duplicates = pd.read_csv(settings.DATA / 'isic-2020-challenge' / 'ISIC_2020_Training_Duplicates.csv')
    df_metadata = df_metadata.loc[~df_metadata['isic_id'].isin(df_isic_2020_duplicates['image_name_2'])]
    df_metadata = df_metadata.loc[~df_metadata['isic_id'].apply(lambda x: 'down' in x)]
    df_metadata = df_metadata.drop_duplicates(subset='isic_id', keep='last').reset_index(drop=True)

    track_stats = False
    if track_stats:
        for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

            image = cv2.imread(row['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            means = np.mean(image, axis=(0, 1))
            df_metadata.loc[idx, 'r_mean'] = means[0]
            df_metadata.loc[idx, 'g_mean'] = means[1]
            df_metadata.loc[idx, 'b_mean'] = means[2]

    duplicate_ids = [
        'ISIC_0001239', 'ISIC_0001241', 'ISIC_0001243', 'ISIC_0001240',
        'ISIC_0055643', 'ISIC_0001244', 'ISIC_0061391', 'ISIC_0057272',
        'ISIC_0013148', 'ISIC_0735344', 'ISIC_6668182', 'ISIC_0067799',
        'ISIC_0053738', 'ISIC_0012536', 'ISIC_0070713', 'ISIC_0057051',
        'ISIC_0057477', 'ISIC_0054118', 'ISIC_0060116', 'ISIC_0054763',
        'ISIC_0054334', 'ISIC_0061121', 'ISIC_0005252', 'ISIC_0014468',
        'ISIC_0058770', 'ISIC_0066634', 'ISIC_0005253', 'ISIC_0061802',
        'ISIC_0022211', 'ISIC_0005538', 'ISIC_0014086', 'ISIC_0001394',
        'ISIC_0061902', 'ISIC_0056728', 'ISIC_0006908', 'ISIC_2258860',
        'ISIC_0005534', 'ISIC_0014039', 'ISIC_0001393', 'ISIC_0063304',
        'ISIC_0012271', 'ISIC_0013083', 'ISIC_0061089', 'ISIC_0001389',
        'ISIC_0062983', 'ISIC_0005537', 'ISIC_0001246', 'ISIC_0013931',
        'ISIC_3386975', 'ISIC_0006873', 'ISIC_0001245', 'ISIC_0021742',
        'ISIC_0001265', 'ISIC_0005533', 'ISIC_3844971', 'ISIC_0013992',
        'ISIC_0001388', 'ISIC_0006622', 'ISIC_0001390', 'ISIC_0013119',
        'ISIC_0012754', 'ISIC_0006767', 'ISIC_0001398', 'ISIC_0053812',
        'ISIC_0001231', 'ISIC_0005535', 'ISIC_0005251', 'ISIC_0014091',
        'ISIC_0001326', 'ISIC_0013273', 'ISIC_0005536', 'ISIC_0001263',
        'ISIC_0012970', 'ISIC_0001382', 'ISIC_0001290', 'ISIC_0006794',
        'ISIC_0005531', 'ISIC_0060482', 'ISIC_0001391', 'ISIC_0001228',
        'ISIC_0005532', 'ISIC_0005246', 'ISIC_0006864', 'ISIC_0001264',
        'ISIC_0013444', 'ISIC_0001294', 'ISIC_0001364', 'ISIC_0023391',
        'ISIC_0013256', 'ISIC_0001430', 'ISIC_0006688', 'ISIC_0001229',
        'ISIC_0001395', 'ISIC_0060997', 'ISIC_0007645', 'ISIC_0001413',
        'ISIC_0001433', 'ISIC_0001386', 'ISIC_0007379', 'ISIC_0001392',
        'ISIC_0012221', 'ISIC_0013377', 'ISIC_0059011', 'ISIC_0001396',
        'ISIC_0001432', 'ISIC_3377854', 'ISIC_0001457', 'ISIC_0007424',
        'ISIC_0012530', 'ISIC_0001288', 'ISIC_0001282', 'ISIC_0001371',
        'ISIC_0001332', 'ISIC_0001441', 'ISIC_0001291', 'ISIC_0006476',
        'ISIC_0005249', 'ISIC_0001285', 'ISIC_0054023', 'ISIC_0001328',
        'ISIC_0006605', 'ISIC_0014398', 'ISIC_0001287', 'ISIC_0001431',
        'ISIC_0023264', 'ISIC_0001268', 'ISIC_0001380', 'ISIC_0013008',
        'ISIC_0060606', 'ISIC_0001230', 'ISIC_0001293', 'ISIC_0001410',
        'ISIC_0001314', 'ISIC_0024777', 'ISIC_0001283', 'ISIC_0006741',
        'ISIC_0012216', 'ISIC_0006601', 'ISIC_0001348', 'ISIC_0001439',
        'ISIC_0001399', 'ISIC_0001277', 'ISIC_0001255', 'ISIC_0001456',
        'ISIC_0013810', 'ISIC_0001383', 'ISIC_0001235', 'ISIC_0001363',
        'ISIC_0001266', 'ISIC_0001289', 'ISIC_0006764', 'ISIC_0001387',
        'ISIC_0001237', 'ISIC_0006452', 'ISIC_0001273', 'ISIC_0007421',
        'ISIC_0001236', 'ISIC_0007527', 'ISIC_0065588', 'ISIC_0434890',
        'ISIC_0001440', 'ISIC_0001378', 'ISIC_0001280', 'ISIC_0007053',
        'ISIC_0001397', 'ISIC_0001271', 'ISIC_0001438', 'ISIC_0006828',
        'ISIC_1950045', 'ISIC_0001385', 'ISIC_0001284', 'ISIC_0063659',
        'ISIC_0001412', 'ISIC_0001411', 'ISIC_0001370', 'ISIC_0001351',
        'ISIC_0061710', 'ISIC_0013331', 'ISIC_0001414', 'ISIC_0001324',
        'ISIC_0001331', 'ISIC_0001233', 'ISIC_0007782', 'ISIC_0001257',
        'ISIC_0001418', 'ISIC_0012745', 'ISIC_0013368', 'ISIC_0559101',
        'ISIC_0033300', 'ISIC_0001384', 'ISIC_0066413', 'ISIC_0001454',
        'ISIC_0001416', 'ISIC_0005250', 'ISIC_0001305', 'ISIC_0001376',
        'ISIC_0001408', 'ISIC_1663074', 'ISIC_0001248', 'ISIC_0001379',
        'ISIC_0001444', 'ISIC_0014553', 'ISIC_0001234', 'ISIC_0001345',
        'ISIC_0924293', 'ISIC_0001256', 'ISIC_0001327', 'ISIC_0001458',
        'ISIC_0001302', 'ISIC_0001330', 'ISIC_0001325', 'ISIC_0001417',
        'ISIC_0023436', 'ISIC_0064300', 'ISIC_0007572', 'ISIC_0001249',
        'ISIC_0013143', 'ISIC_0006760', 'ISIC_0001252', 'ISIC_0001329',
        'ISIC_0007264', 'ISIC_0001274', 'ISIC_0001272', 'ISIC_0001259',
        'ISIC_0059661', 'ISIC_0001251', 'ISIC_0001261', 'ISIC_0061437',
        'ISIC_0064007', 'ISIC_0064670', 'ISIC_0006576', 'ISIC_0001253',
        'ISIC_0001373', 'ISIC_0001276', 'ISIC_0013011', 'ISIC_0001346',
        'ISIC_0054792', 'ISIC_0001459', 'ISIC_0001424', 'ISIC_0012977',
        'ISIC_0001269', 'ISIC_0014010', 'ISIC_0059354', 'ISIC_0001316',
        'ISIC_0054074', 'ISIC_0054113', 'ISIC_0069013', 'ISIC_0001361',
        'ISIC_0001319', 'ISIC_0001270', 'ISIC_0001443', 'ISIC_0062259',
        'ISIC_0001400', 'ISIC_0001448', 'ISIC_0001214', 'ISIC_0001258',
        'ISIC_0001406', 'ISIC_0001375', 'ISIC_0054075', 'ISIC_0001279',
        'ISIC_0001309', 'ISIC_0007434', 'ISIC_0001402', 'ISIC_0001315',
        'ISIC_0001452', 'ISIC_2568903', 'ISIC_0001381', 'ISIC_0001415',
        'ISIC_0001442', 'ISIC_0001281', 'ISIC_0001318', 'ISIC_0001407',
        'ISIC_0006785', 'ISIC_0001232', 'ISIC_0001278', 'ISIC_0013916',
        'ISIC_0001409', 'ISIC_0001368', 'ISIC_0013161', 'ISIC_0001260',
        'ISIC_0013744', 'ISIC_0001250', 'ISIC_0067502', 'ISIC_0013771',
        'ISIC_0001450', 'ISIC_0001202', 'ISIC_0001200', 'ISIC_0001369',
        'ISIC_0001349', 'ISIC_0001451', 'ISIC_0001337', 'ISIC_0001425',
        'ISIC_0001344', 'ISIC_0065140', 'ISIC_0001217', 'ISIC_0001350',
        'ISIC_0001347', 'ISIC_0001405', 'ISIC_0001455', 'ISIC_0001338',
        'ISIC_0001377', 'ISIC_0001334', 'ISIC_0001419', 'ISIC_0001304',
        'ISIC_0001401', 'ISIC_0001312', 'ISIC_0001453', 'ISIC_0001317',
        'ISIC_0001201', 'ISIC_0001198', 'ISIC_0001183', 'ISIC_0001303',
        'ISIC_0001298', 'ISIC_0001300', 'ISIC_0001301', 'ISIC_0013959',
        'ISIC_0001366', 'ISIC_0001403', 'ISIC_0001362', 'ISIC_0001422',
        'ISIC_0001404', 'ISIC_0001420', 'ISIC_0001365', 'ISIC_0001447',
        'ISIC_0001341', 'ISIC_0001335', 'ISIC_0001340', 'ISIC_0001295',
        'ISIC_0001339', 'ISIC_0000279', 'ISIC_0001199', 'ISIC_0001184',
        'ISIC_0001445', 'ISIC_0012948', 'ISIC_0001336', 'ISIC_0001218',
        'ISIC_0001176', 'ISIC_0001343', 'ISIC_0001429', 'ISIC_0001342',
        'ISIC_0001189', 'ISIC_0001297', 'ISIC_0001203', 'ISIC_0001310',
        'ISIC_0001428', 'ISIC_0001219', 'ISIC_0014175', 'ISIC_0001238',
        'ISIC_0001210', 'ISIC_0001209', 'ISIC_0001446', 'ISIC_0001357',
        'ISIC_0014235', 'ISIC_0013960', 'ISIC_0013093', 'ISIC_0001333',
        'ISIC_0001311', 'ISIC_0001313', 'ISIC_0001307', 'ISIC_0001308',
        'ISIC_0014152', 'ISIC_0022209', 'ISIC_0001421', 'ISIC_0001182',
        'ISIC_0054911', 'ISIC_0001435', 'ISIC_0013019', 'ISIC_0013381',
        'ISIC_0061059', 'ISIC_0001358', 'ISIC_0012771', 'ISIC_0022210',
        'ISIC_0001211', 'ISIC_0013312', 'ISIC_0054986', 'ISIC_0001434',
        'ISIC_0001215', 'ISIC_0001225', 'ISIC_0023553', 'ISIC_0001354',
        'ISIC_0001197', 'ISIC_0001194', 'ISIC_0012935', 'ISIC_0001193',
        'ISIC_0001360', 'ISIC_0001178', 'ISIC_0013607', 'ISIC_0001205',
        'ISIC_0001226', 'ISIC_0062361', 'ISIC_0001177', 'ISIC_0001166',
        'ISIC_0013095', 'ISIC_0001192', 'ISIC_0001426', 'ISIC_0001359',
        'ISIC_0001223', 'ISIC_0012152', 'ISIC_0001206', 'ISIC_0001437',
        'ISIC_0001169', 'ISIC_0001355', 'ISIC_0012175', 'ISIC_0001227',
        'ISIC_0013872', 'ISIC_0001167', 'ISIC_0001353', 'ISIC_0001179',
        'ISIC_8166745', 'ISIC_0001224', 'ISIC_0013298', 'ISIC_1642492',
        'ISIC_0001162', 'ISIC_2313463', 'ISIC_0001208', 'ISIC_0001221',
        'ISIC_0001352', 'ISIC_0001175', 'ISIC_0001356', 'ISIC_0001220',
        'ISIC_0013135', 'ISIC_0001171', 'ISIC_0001195', 'ISIC_0001222',
        'ISIC_0014584', 'ISIC_0001160', 'ISIC_0001173', 'ISIC_0001165',
        'ISIC_0001172', 'ISIC_0001180', 'ISIC_0001170', 'ISIC_0001164',
        'ISIC_0012997', 'ISIC_3656810', 'ISIC_5746049', 'ISIC_0001174',
        'ISIC_3588900', 'ISIC_0012843', 'ISIC_2376329', 'ISIC_0001436',
        'ISIC_6063252', 'ISIC_0001161', 'ISIC_0001196', 'ISIC_8061920',
        'ISIC_0506082', 'ISIC_6275427', 'ISIC_1026838', 'ISIC_0001168',
        'ISIC_0001207', 'ISIC_3083302', 'ISIC_2718135', 'ISIC_2394733',
        'ISIC_0022212', 'ISIC_0023833', 'ISIC_4308112', 'ISIC_0762091',
        'ISIC_5265808', 'ISIC_0517531', 'ISIC_5193073', 'ISIC_3508787',
        'ISIC_2862111', 'ISIC_7607101', 'ISIC_1142863', 'ISIC_4156931',
        'ISIC_0001320', 'ISIC_6249626', 'ISIC_5492174', 'ISIC_0001322',
        'ISIC_4061347', 'ISIC_1270937', 'ISIC_7005958', 'ISIC_0022208',
        'ISIC_4990272', 'ISIC_2074396', 'ISIC_3275875', 'ISIC_7480177',
        'ISIC_0012899', 'ISIC_3276396', 'ISIC_4294619', 'ISIC_1848959',
        'ISIC_0024366', 'ISIC_1115450', 'ISIC_1437627', 'ISIC_3796609',
        'ISIC_3716403', 'ISIC_2131119', 'ISIC_6209867', 'ISIC_2817101',
        'ISIC_0704435', 'ISIC_0001323', 'ISIC_4829472', 'ISIC_4495527',
        'ISIC_0001321', 'ISIC_0635396', 'ISIC_6284722', 'ISIC_1171499',
        'ISIC_0013770', 'ISIC_1425639', 'ISIC_4443268', 'ISIC_6248112',
        'ISIC_3046977', 'ISIC_1706022', 'ISIC_4247203', 'ISIC_5467408',
        'ISIC_4545487', 'ISIC_6450285', 'ISIC_0013255', 'ISIC_2689674',
        'ISIC_4019458', 'ISIC_6170663', 'ISIC_8889010', 'ISIC_3004197',
        'ISIC_4537621', 'ISIC_2890008', 'ISIC_9008728', 'ISIC_2757628',
        'ISIC_4892807', 'ISIC_2964421', 'ISIC_2200299', 'ISIC_5006966',
        'ISIC_2391447', 'ISIC_2507875', 'ISIC_8683473', 'ISIC_2192031',
        'ISIC_4819603', 'ISIC_5866452', 'ISIC_2190511'
    ]
    #df_metadata = df_metadata.loc[~df_metadata['isic_id'].isin(duplicate_ids)].reset_index(drop=True)
    df_metadata = df_metadata.drop_duplicates(subset=['r_mean', 'g_mean', 'b_mean'], keep='last')
    settings.logger.info(
        f'''
        ISIC Datasets Target Counts
        {df_metadata.groupby('dataset')['target'].value_counts()}
        '''
    )

    isic_other = [
        'prove-ai', 'consecutive-biopsies-2020', 'consumer-ai-apps',
        'easy-dermoscopy-expert-agreement-study', 'hiba-skin-lesions',
        'longitudinal-images-with-various-types',
        'melanoma-and-nevus-dermoscopy-images', 'msk-1', 'msk-2', 'msk-3',
        'msk-4', 'msk-5', 'newly-acquired-melanoma',
        'repeated-dermoscopic-images', 'sonic', 'uda-1', 'uda-2',
        'isic_2016', 'isic_2017'
    ]
    df_metadata.loc[df_metadata['dataset'].isin(isic_other), 'dataset'] = 'isic_other'

    df_metadata.to_parquet(output_directory / 'isic_metadata.parquet')
    settings.logger.info(f'isic_metadata.parquet is saved to {output_directory}')
