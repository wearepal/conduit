from __future__ import annotations
from pathlib import Path

from kit import implements
from torch import Tensor

from bolts.data.datasets.base import PBDataset
from bolts.data.datasets.utils import (
    ImageLoadingBackend,
    ImageTform,
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
        y: TargetData | None = None,
        s: TargetData | None = None,
        transform: ImageTform | None = None,
        root: Path | str | None = None,
    ) -> None:
        super().__init__(x=x, y=y, s=s)
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.transform = transform
        # infer the appropriate image-loading backend based on the type of 'transform'
        self._il_backend: ImageLoadingBackend = infer_il_backend(self.transform)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @implements(PBDataset)
    def _sample_x(self, index: int) -> Tensor:
        image = load_image(self._base_dir / self.x[index], backend=self._il_backend)
        image = apply_image_transform(image=image, transform=self.transform)
        if not isinstance(image, Tensor):
            image = img_to_tensor(image)
        return image
