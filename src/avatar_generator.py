import io
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid

from dalle_pytorch import VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from src.data_processing.helpers import AvatarConfigProperty


def form_caption(avatar_config: dict) -> str:
    _IGNORE_PROPS = ["style", "nose_type", "clothe_graphic_type"]
    caption = ""

    for prop in AvatarConfigProperty:
        if prop.value in _IGNORE_PROPS or prop.value not in avatar_config:
            continue

        caption += f"{prop.value.replace('_', ' ')} {avatar_config[prop.value]}, "

    # Remove last comma.
    caption = caption[:-2]
    return caption


def pil_2_bytes(image: Image):
    with io.BytesIO() as out:
        image.save(out, format="PNG")
        contents = out.getvalue()

    return contents


class AvatarGenerator:
    __DALLE_SAMPLES = 1
    __DALLE_PATH = Path("./weights/best/dalle.pt")

    def __init__(self):
        self.__dalle = self.__load_dalle()

    def __load_dalle(self):
        assert self.__DALLE_PATH.exists(), 'Incorrect path for DALL-E weights!'
        load_obj = torch.load(str(self.__DALLE_PATH))
        dalle_params = load_obj.pop('hparams')
        weights = load_obj.pop('weights')

        vae = VQGanVAE()
        dalle = DALLE(vae=vae, **dalle_params).cuda()
        dalle.load_state_dict(weights)
        return dalle

    def __infer_dalle(self, caption: str) -> Tensor:
        text_tokens = tokenizer.tokenize([caption], self.__dalle.text_seq_len).cuda()
        output = self.__dalle.generate_images(text_tokens, filter_thres=0.9)
        return output

    def __generate_avatars(self, caption: str) -> Tensor:
        dalle_generated_samples = self.__infer_dalle(caption)
        return dalle_generated_samples

    @staticmethod
    def __samples_postprocessing(samples: Tensor) -> Image:
        outputs = list()

        for sample in samples:
            grid = make_grid(sample, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            outputs.append(im)

        return outputs

    def generate_avatars(self, caption: str):
        samples = self.__generate_avatars(caption)
        samples = self.__samples_postprocessing(samples)
        return samples
