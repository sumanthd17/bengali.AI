import pandas as pd
import joblib
from tqdm import tqdm
import glob

if __name__=="__main__":
    files = glob.glob('../input_files/train_*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        images_ids = df.image_id.values
        df = df.drop('image_id', axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(images_ids), total=len(images_ids)):
            joblib.dump(image_array[j, :], '../input_files/image_pickles/{}.pkl'.format(img_id))