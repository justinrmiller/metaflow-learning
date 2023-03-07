import time

import torch

import glob
import requests
import zipfile

import os

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from PIL import Image

from transformers import CLIPModel, AutoProcessor

from metaflow import FlowSpec, step


class EmbeddingsFlow(FlowSpec):
    def chunks(self, list, n):
        """ Yield successive n-sized chunks from list.
        """
        for i in range(0, len(list), n):
            yield list[i:i + n]

    @step
    def start(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model_name = "openai/clip-vit-base-patch32"
        # self.model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

        self.max_number_of_images = 3000
        self.modulo = 10
        self.chunk_size = 20

        path = "val2017/*.jpg"
        zip_url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_filename = "val2017.zip"

        if not os.path.exists(zip_filename):
            # If it doesn't exist, download the zip file
            print(f"f{zip_filename} does not exist, downloading...")
            r = requests.get(zip_url, stream=True)
            with open(zip_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f'{zip_filename} downloaded.')
        else:
            print(f'{zip_filename} already exists.')

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('.')
            print(f'{zip_filename} extracted.')

        jpg_files = glob.glob(path)

        self.image_paths = jpg_files[:self.max_number_of_images]
        self.number_of_embeddings = len(self.image_paths)

        self.parquet_file_name = "output.snappy.parquet"

        self.next(self.generate)

    @step
    def generate(self):
        print(
            "Model loading..."
            f" device used: {self.device}"
            f" number of embeddings to generate: {self.number_of_embeddings}"
            f" modulo: {self.modulo}"
        )

        model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)

        start_time = time.perf_counter()

        embeddings = []

        chunked_image_paths = self.chunks(self.image_paths, self.chunk_size)

        processed_count = 0

        for chunk in chunked_image_paths:
            images = [Image.open(image_path) for image_path in chunk]

            inputs = processor(images=images, return_tensors="pt").to(self.device)

            image_features_for_partition = model.get_image_features(**inputs)

            for index, image_features in enumerate(image_features_for_partition):
                name = chunk[index].split("/")[1]
                embeddings.append([name, image_features.detach().cpu().numpy()])

            processed_count += len(image_features_for_partition)

            if processed_count % self.modulo == 0:
                print(f"Processed images: {processed_count}")
            
            for image in images:
                image.close()
                del image
            del images

        end_time = time.perf_counter()
        time_taken = end_time-start_time

        print(
            f"Time taken: {time_taken}, "
            f"Encodings per second: {self.number_of_embeddings/time_taken}"
        )

        df = pd.DataFrame(embeddings, columns=['name', 'embedding'], dtype=float)

        arrow_table = pa.Table.from_pandas(df)

        pq.write_table(
            arrow_table,
            self.parquet_file_name,
            use_dictionary=False
        )

        self.next(self.end)

    @step
    def end(self):
        print(f"Embeddings generated. Parquet file w/Snappy compression available here: {self.parquet_file_name}")


if __name__ == '__main__':
    EmbeddingsFlow()
