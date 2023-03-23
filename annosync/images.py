import os
import pathlib
import warnings
from typing import Tuple

import cv2
import cytomine
import cytomine.models as cm
import numpy as np
import PIL.Image
import shapely
import shapely.errors
import shapely.wkt
import sldc
from shapely.geometry.point import Point


class CytominePIMSTile(sldc.Tile):
    """FIX for newer version of Cytomine. Adapted from
    https://github.com/bathienle/S_Create_Annotations/blob/b7a0d6916d540d6fde8c74435ce18e5e2e2695cd/run.py#L43"""

    def __init__(
        self,
        working_path,
        parent,
        offset,
        width,
        height,
        tile_identifier=None,
        polygon_mask=None,
        n_jobs=1,
    ):
        super().__init__(
            parent,
            offset,
            width,
            height,
            tile_identifier=tile_identifier,
            polygon_mask=polygon_mask,
        )

        self._working_path = working_path
        self._n_jobs = n_jobs
        os.makedirs(working_path, exist_ok=True)

    @property
    def cache_filename(self):
        image_instance = self.base_image.image_instance
        x, y = self.abs_offset_x, self.abs_offset_y
        width, height = self.width, self.height
        zoom = self.base_image.zoom_level
        return f"{image_instance.id}-{zoom}-{x}-{y}-{width}-{height}.png"

    @property
    def cache_filepath(self):
        return os.path.join(self._working_path, self.cache_filename)

    @property
    def np_image(self):
        try:
            if (
                not os.path.exists(self.cache_filepath)
                and not self.download_tile_image()
            ):
                raise sldc.TileExtractionException(
                    f"Cannot fetch tile at for '{self.cache_filename}'."
                )

            np_array = np.asarray(PIL.Image.open(self.cache_filepath)).squeeze()

            if (
                np_array.shape[:2] != (self.height, self.width)
                or (
                    self.channels > 1
                    and (np_array.ndim < 3 or np_array.shape[2] != self.channels)
                )
                or (
                    self.channels == 1
                    and np_array.ndim > 2
                    and np_array.shape[2] != self.channels
                )
            ):
                raise sldc.TileExtractionException(
                    f"Fetched image has invalid size : {np_array.shape} instead "
                    f"of {(self.width, self.height, self.channels)}"
                )

            if np_array.ndim == 3 and np_array.shape[2] == 4:
                np_array = np_array[:, :, :3]

            return np_array.astype("uint8")
        except IOError as e:
            raise sldc.TileExtractionException(str(e))

    def download_tile_image(self):
        slide = self.base_image
        filepath: str = slide.image_instance.path
        topology = sldc.TileTopology(slide, None, max_width=256, max_height=256)
        col_tile: int = self.abs_offset_x // 256
        row_tile: int = self.abs_offset_y // 256
        tile_index: int = col_tile + row_tile * topology.tile_horizontal_count
        _slice: cm.ImageInstance = slide.slice_instance

        url = (
            f"{_slice.imageServerUrl}/image/{filepath}/tile/"
            f"zoom/{slide.api_zoom_level}/ti/{tile_index}.png"
        )

        return cytomine.Cytomine.get_instance().download_file(url, self.cache_filepath)


def image_shape(path: pathlib.Path) -> Tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"{path=!r} does not exist")

    return img.shape[1::-1]


def fix_grayscale(path: pathlib.Path):
    "save an RGB grayscale image as single channel"

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{path=!r} does not exist")

    # encoded as grayscale: no need to worry about it
    if img.shape[-1] == 1:
        return

    if img.shape[-1] != 3:
        warnings.warn(
            f"fix grayscale {path=!r}: expected RGB, "
            f"found {img.shape[-1]} channels. Skipping"
        )
        return

    # compute the median value for each channel
    channel_median = np.median(img.reshape((-1, img.shape[-1])), axis=0)
    channel_peak_diff = channel_median.max() - channel_median.min()

    val_range = img.max() - img.min()

    # actual color image
    if channel_peak_diff > 0.05 * val_range:
        return

    # store as grayscale
    path.unlink()
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def iou_annotations(
    left: cm.Annotation,
    right: cm.Annotation,
) -> float:
    "compute IoU for two annotations"

    geometry_l = shapely.wkt.loads(left.location)
    geometry_r = shapely.wkt.loads(right.location)

    if geometry_l.area == 0:
        raise TypeError(f"2D geometry required for {geometry_l=!r}")
    if geometry_r.area == 0:
        raise TypeError(f"2D geometry required for {geometry_r=!r}")

    inter = geometry_l.intersection(geometry_r).area
    union = geometry_l.area + geometry_r.area - inter

    return inter / union


def dice_annotations(
    left: cm.Annotation,
    right: cm.Annotation,
) -> float:
    "compute the dice score for two annotations"

    geometry_l = shapely.wkt.loads(left.location)
    geometry_r = shapely.wkt.loads(right.location)

    if geometry_l.area == 0:
        raise TypeError(f"2D geometry required for {geometry_l=!r}")
    if geometry_r.area == 0:
        raise TypeError(f"2D geometry required for {geometry_r=!r}")

    inter = geometry_l.intersection(geometry_r).area

    return 2.0 * inter / (geometry_l.area + geometry_r.area)


def distance_annotations(
    left: cm.Annotation,
    right: cm.Annotation,
) -> float:
    "computes the mean TRE (L2 distance) for the points in two annotations"

    geometry_l = shapely.wkt.loads(left.location)
    geometry_r = shapely.wkt.loads(right.location)

    return geometry_l.distance(geometry_r)


def tre_annotations(
    left: cm.Annotation,
    right: cm.Annotation,
) -> float:
    "compute TRE (L2 distance) for two Point annotations"

    geometry_l = shapely.wkt.loads(left.location)
    geometry_r = shapely.wkt.loads(right.location)

    if not isinstance(geometry_l, Point):
        raise TypeError(f"Point required for {geometry_l=!r}")
    if not isinstance(geometry_r, Point):
        raise TypeError(f"Point required for {geometry_r=!r}")

    return geometry_l.distance(geometry_r)
