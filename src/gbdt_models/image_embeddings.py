import sys
from tqdm import tqdm
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


sys.path.append('..')
import settings



if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    model_directory = settings.MODELS / 'dinov2' / 'dinov2-pytorch-small-v1'
    device = torch.device('cuda')
    processor = AutoImageProcessor.from_pretrained(model_directory)
    model = AutoModel.from_pretrained(model_directory).eval().to(device)

    train_images = h5py.File(settings.DATA / 'isic-2024-challenge' / 'train-image.hdf5', 'r+')

    image_embeddings = []

    for isic_id in tqdm(train_images):

        image = np.array(Image.open(BytesIO(train_images[isic_id][()])))

        inputs = processor(images=image, return_tensors='pt', do_rescale=False).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(**inputs)

        outputs = outputs.last_hidden_state.cpu()[:, 1:].max(dim=1)[0]
        outputs = F.normalize(outputs, dim=-1, p=2)

        image_embeddings.append(outputs)

    image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
    np.save(settings.DATA / 'image_embeddings.npy', image_embeddings)
