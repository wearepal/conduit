from __future__ import annotations
from pathlib import Path

from kit import implements
from torch import Tensor

from bolts.data.datasets.base import PBDataset
from bolts.data.datasets.utils import (
    ImageLoadingBackend,
    ImageTform,
    RawImage,
    apply_image_transform,
    img_to_tensor,
    infer_il_backend,
    load_image,
)
from bolts.data.structures import InputData, TargetData

__all__ = ["PBVisionDataset"]


class PBVisionDataset(PBDataset):
    def __init__(
        self,
        x: InputData,
        image_dir: Path | str,
        y: TargetData | None = None,
        s: TargetData | None = None,
        transform: ImageTform | None = None,
    ) -> None:
        super().__init__(x=x, y=y, s=s)
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.transform = transform
        # infer the appropriate image-loading backend based on the type of 'transform'
        self._il_backend: ImageLoadingBackend = infer_il_backend(self.transform)
        self.log(f"Using {self._il_backend} as backend for image-loading.")

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        body.append(f"Base image-directory location: {self.image_dir.resolve()}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _load_image(self, index: int) -> RawImage:
        return load_image(self.image_dir / self.x[index], backend=self._il_backend)

    @implements(PBDataset)
    def _sample_x(self, index: int, coerce_to_tensor: bool = False) -> RawImage | Tensor:
        image = self._load_image(index)
        image = apply_image_transform(image=image, transform=self.transform)
        if coerce_to_tensor and (not isinstance(image, Tensor)):
            image = img_to_tensor(image)
        return image
