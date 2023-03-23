import contextlib
import enum
import os
from typing import Any, Dict, Iterable, Literal, Mapping, Union

import cytomine.models as cm


def pretty_repr(o: Any) -> str:
    if isinstance(o, (int, float, str, type(None))):
        return f"{o!r}"

    if isinstance(o, enum.Enum):
        return type(o).__name__ + "." + o.name

    # named tuple
    if isinstance(o, tuple) and hasattr(o, "_asdict"):
        return f"{o!r}"

    if isinstance(o, Mapping):
        inner = ", ".join(pretty_repr(k) + ":" + pretty_repr(v) for k, v in o.items())
        return "{" + inner + "}"

    if isinstance(o, Iterable):
        return "[" + ", ".join(pretty_repr(s) for s in o) + "]"

    if hasattr(o, "__dict__"):
        return str(type(o)) + ":" + pretty_repr(vars(o))

    if hasattr(o, "__slots__"):
        inner = pretty_repr({attr: getattr(o, attr) for attr in o.__slots__})
        return str(type(o)) + ":" + inner

    # default representation
    return f"{o!r}"


_img_group_cache: Dict[int, cm.ImageGroup] = {}
_img_col_cache: Dict[int, cm.ImageInstanceCollection] = {}
_img_cache: Dict[int, cm.ImageInstance] = {}


def _fetch_image_group(
    image_group: int,
) -> Union[cm.ImageGroup, Literal[False]]:
    if ret := _img_group_cache.get(image_group, False):
        return ret

    img_group = cm.ImageGroup().fetch(image_group)
    if img_group:
        _img_group_cache[image_group] = img_group
    return img_group


def _fetch_image_col(
    image_group: int,
) -> Union[cm.ImageInstanceCollection, Literal[False]]:
    "caching only successful responses"
    if ret := _img_col_cache.get(image_group, False):
        return ret

    img_col = cm.ImageInstanceCollection().fetch_with_filter("imagegroup", image_group)
    if img_col:
        _img_col_cache[image_group] = img_col
    return img_col


def _fetch_image(
    image: int,
) -> Union[cm.ImageInstance, Literal[False]]:
    if ret := _img_cache.get(image, False):
        return ret

    img = cm.ImageInstance().fetch(image)
    if img:
        _img_cache[image] = img
    return img


@contextlib.contextmanager
def no_output(silent: bool):
    try:
        if not silent:
            yield None
            return
        with open(os.devnull, "w", encoding="utf8") as devnull:
            with contextlib.redirect_stderr(devnull):
                with contextlib.redirect_stdout(devnull):
                    yield None
    finally:
        pass
