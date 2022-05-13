import io

import numpy as np
from PIL import Image
from torch import Tensor


def form_caption(avatar_config: dict) -> str:
    return "caption"


def pil_2_bytes(image: Image):
    with io.BytesIO() as out:
        image.save(out, format="PNG")
        contents = out.getvalue()

    return contents


# TODO: make it singleton avoid creating new object every call. Or google, how to do it correctly.
class AvatarGenerator:
    __DALLE_SAMPLES = 32
    __AVATAR_RESOLUTION = (256, 256)

    def __infer_dalle(self, caption: str) -> Tensor:
        pass

    def __clip_rerank(self, candidate_samples: Tensor, caption: str, n_samples: int) -> Tensor:
        pass

    def __generate_avatars(self, caption: str, n_samples: int) -> Tensor:
        dalle_generated_samples = self.__infer_dalle(caption)
        best_generated_samples = self.__clip_rerank(dalle_generated_samples, caption, n_samples)

        return best_generated_samples

    def __samples_postprocessing(self, samples: Tensor) -> Image:
        pass

    def __dummy_result(self, n_samples: int):
        return [Image.fromarray(np.ones(self.__AVATAR_RESOLUTION) * 127, mode="RGB")] * n_samples

    def generate_avatars(self, caption: str, n_samples: int = 4):
        samples = self.__dummy_result(n_samples)
        return samples


