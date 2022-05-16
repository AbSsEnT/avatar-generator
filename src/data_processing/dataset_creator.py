import os
import re
from uuid import uuid4

import numpy as np
import shutil as sh
from tqdm import tqdm
import py_avataaars as pa
from numpy.random import choice
from py_avataaars import PyAvataaar
from joblib import Parallel, delayed


class DatasetCreator:
    @staticmethod
    def __pick_random(prop: pa.AvatarEnum):
        return choice(list(prop))

    def __get_random_avatar(self):
        avatar_config = {
            "style": pa.AvatarStyle.TRANSPARENT,
            "skin_color": self.__pick_random(pa.SkinColor),
            "hair_color": self.__pick_random(pa.HairColor),
            "facial_hair_type": self.__pick_random(pa.FacialHairType),
            "facial_hair_color": self.__pick_random(pa.HairColor),
            "top_type": self.__pick_random(pa.TopType),
            "hat_color": self.__pick_random(pa.Color),
            "mouth_type": self.__pick_random(pa.MouthType),
            "eye_type": self.__pick_random(pa.EyesType),
            "nose_type": pa.NoseType.DEFAULT,
            "eyebrow_type": self.__pick_random(pa.EyebrowType),
            "accessories_type": self.__pick_random(pa.AccessoriesType),
            "clothe_type": self.__pick_random(pa.ClotheType),
            "clothe_color": self.__pick_random(pa.Color),
        }

        return PyAvataaar(**avatar_config), avatar_config

    @staticmethod
    def __process_word(word: str):
        word = word.lower()
        word = word.replace("_", " ")

        match = re.search("\d+", word)
        if match is not None:
            drop_from = match.start()
            word = word[:drop_from]

        return word

    def __config_2_caption(self, avatar_config: dict):
        _IGNORE_PROPS = ["style", "nose_type", "clothe_graphic_type"]
        caption = ""

        for key, value in avatar_config.items():
            if key in _IGNORE_PROPS:
                continue

            caption += f"{self.__process_word(key)} {self.__process_word(str(value))}, "

        # Remove last comma.
        caption = caption[:-2]
        return caption

    @staticmethod
    def __write_txt(path: str, caption: str):
        with open(path, "w") as out_file:
            out_file.write(caption)

    @staticmethod
    def __prepare_save_path(save_path: str):
        if os.path.exists(save_path):
            sh.rmtree(save_path)

        os.makedirs(save_path)

    def generate_sample(self, save_path: str, random_state: int or None):
        np.random.seed(random_state)
        filename = uuid4()
        avatar, avatar_config = self.__get_random_avatar()
        avatar_caption = self.__config_2_caption(avatar_config)

        # Save generated sample.
        avatar.render_png_file(os.path.join(save_path, f"{filename}.png"))
        self.__write_txt(os.path.join(save_path, f"{filename}.txt"), avatar_caption)

    def create_dataset(self, save_path: str, n_samples: int = 10 ** 5):
        print(f"Starting to generate {n_samples} samples.")
        print(f"Saving data to {save_path} directory.")

        self.__prepare_save_path(save_path)

        Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.generate_sample)(save_path, np.random.randint(0, 100000)) for _ in tqdm(range(n_samples)))
