import base64
import io
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image

from app.ml.diffusion import GaussianDiffusion
from app.ml.model import UNet


def _tensor_to_base64_png(img: torch.Tensor) -> str:
    array = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Generator(ABC):
    @abstractmethod
    def generate(self, num_samples: int, seed: Optional[int] = None) -> List[str]:
        """Return `num_samples` base64-encoded PNGs."""


class MockGenerator(Generator):
    """Random noise, no model load, no GPU. Lets the API layer (job
    lifecycle, validation, HTTP contract) be tested in CI without a trained
    checkpoint or an accelerator.
    """

    def generate(self, num_samples: int = 1, seed: Optional[int] = None) -> List[str]:
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        images = torch.rand(num_samples, 3, 32, 32, generator=rng)
        return [_tensor_to_base64_png(img) for img in images]


class DiffusionGenerator(Generator):
    """Loads a checkpoint trained by github.com/Shivam101s/mini-diffusion
    and samples from it with DDIM.
    """

    def __init__(self, checkpoint_path: str, ddim_steps: int = 50, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = UNet(base_channels=checkpoint["args"]["base_channels"]).to(self.device)
        self.model.load_state_dict(checkpoint["ema"])
        self.model.eval()

        self.diffusion = GaussianDiffusion(
            timesteps=checkpoint["args"]["timesteps"], device=self.device
        )
        self.ddim_steps = ddim_steps

    def generate(self, num_samples: int = 1, seed: Optional[int] = None) -> List[str]:
        if seed is not None:
            torch.manual_seed(seed)
        samples = self.diffusion.ddim_sample(
            self.model, (num_samples, 3, 32, 32), self.device, ddim_steps=self.ddim_steps
        )
        samples = (samples.clamp(-1, 1) + 1) / 2
        return [_tensor_to_base64_png(img) for img in samples]


def build_generator(kind: str, checkpoint_path: str, ddim_steps: int) -> Generator:
    if kind == "mock":
        return MockGenerator()
    if kind == "diffusion":
        return DiffusionGenerator(checkpoint_path, ddim_steps=ddim_steps)
    raise ValueError(f"unknown generator kind: {kind}")
