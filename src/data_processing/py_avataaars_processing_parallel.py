import os
import re
import numpy as np
from numpy.random import choice
import shutil as sh
from uuid import uuid4
from argparse import ArgumentParser

from tqdm import tqdm
import py_avataaars as pa
from py_avataaars import PyAvataaar
from joblib import Parallel, delayed


def pick_random(prop: pa.AvatarEnum):
    return choice(list(prop))


def get_random_avatar():
    avatar_config = {
        "style": pa.AvatarStyle.TRANSPARENT,
        "skin_color": pick_random(pa.SkinColor),
        "hair_color": pick_random(pa.HairColor),
        "facial_hair_type": pick_random(pa.FacialHairType),
        "facial_hair_color": pick_random(pa.HairColor),
        "top_type": pick_random(pa.TopType),
        "hat_color": pick_random(pa.Color),
        "mouth_type": pick_random(pa.MouthType),
        "eye_type": pick_random(pa.EyesType),
        "nose_type": pa.NoseType.DEFAULT,
        "eyebrow_type": pick_random(pa.EyebrowType),
        "accessories_type": pick_random(pa.AccessoriesType),
        "clothe_type": pick_random(pa.ClotheType),
        "clothe_color": pick_random(pa.Color),
    }

    return PyAvataaar(**avatar_config), avatar_config


def process_word(word: str):
    word = word.lower()
    word = word.replace("_", " ")

    match = re.search("\d+", word)
    if match is not None:
        drop_from = match.start()
        word = word[:drop_from]

    return word


def config_2_caption(avatar_config: dict):
    _IGNORE_PROPS = ["style", "nose_type", "clothe_graphic_type"]
    caption = ""

    for key, value in avatar_config.items():
        if key in _IGNORE_PROPS:
            continue

        caption += f"{process_word(key)} {process_word(str(value))}, "

    # Remove last comma.
    caption = caption[:-2]
    return caption


def write_txt(path: str, caption: str):
    with open(path, "w") as out_file:
        out_file.write(caption)


def prepare_save_path(save_path: str):
    if os.path.exists(save_path):
        sh.rmtree(save_path)

    os.makedirs(save_path)


def generate_sample(save_path: str, random_state: int or None):
    np.random.seed(random_state)
    filename = uuid4()
    avatar, avatar_config = get_random_avatar()
    avatar_caption = config_2_caption(avatar_config)

    # Save generated sample.
    avatar.render_png_file(os.path.join(save_path, f"{filename}.png"))
    write_txt(os.path.join(save_path, f"{filename}.txt"), avatar_caption)


def create_dataset(save_path: str, n_samples: int = 10 ** 5):
    print(f"Starting to generate {n_samples} samples.")
    print(f"Saving data to {save_path} directory.")

    prepare_save_path(save_path)

    Parallel(n_jobs=-1, backend="multiprocessing")(delayed(generate_sample)(save_path, np.random.randint(0, 100000)) for _ in tqdm(range(n_samples)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to generate dataset of avatars with captions.")
    parser.add_argument("--save_path", type=str, help="Path to save generated dataset.")
    parser.add_argument("--n_samples", type=int, help="How many samples to generate.")

    args = parser.parse_args()
    create_dataset(**vars(args))
