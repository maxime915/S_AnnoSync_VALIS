import enum
import warnings
from collections import defaultdict
from typing import (
    Callable,
    DefaultDict,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    overload,
)

import cytomine.models as cm
from valis import registration

from .utils import _fetch_image_col, _fetch_image_group, pretty_repr

T = TypeVar("T")
U = TypeVar("U")


class ImageOrdering(enum.Enum):
    AUTO = "auto"
    NAME = "name"
    CREATED = "created"


class ImageCrop(enum.Enum):
    REFERENCE = "reference"
    ALL = "all"
    OVERLAP = "overlap"


class RegistrationType(enum.Enum):
    RIGID = "rigid"
    NON_RIGID = "non-rigid"
    MICRO = "micro"


def ei(val: str) -> int:
    "expect int"
    if isinstance(val, int):
        return int(val)  # make sure bool values are converted to int
    if not isinstance(val, str):
        raise TypeError(f"expected str, found {type(val)=}")
    if isinstance(val, str) and str(int(val)) == val:
        return int(val)
    raise ValueError(f"{val=!r} is not an int")


def eil(val: str) -> List[int]:
    val = val.strip(" '\"")
    if not val:
        return []  # allow empty lists
    return [ei(v) for v in val.split(",")]


def eb(val: str) -> bool:
    "expect bool"
    if isinstance(val, bool):
        return val
    if not isinstance(val, str):
        raise TypeError(f"expected str, found {type(val)=}")

    val = val.lower().strip(" '\"")
    if val in ["true", "1", "yes"]:
        return True
    if val in ["false", "0", "no"]:
        return False

    raise ValueError(f"{val=!r} is not a bool")


@overload
def get(namespace, key: str, type_: Callable[[str], T], default: U) -> Union[T, U]:
    ...


@overload
def get(namespace, key: str, type_: Callable[[str], T]) -> Union[T, None]:
    ...


def get(
    namespace,
    key: str,
    type_: Callable[[str], T],
    default: Optional[U] = None,
):
    "reads a parameter from the Namespace object, and cast it to the right type"
    ret = getattr(namespace, key, None)
    if ret is None:
        return default
    if isinstance(ret, (str, int, bool, float)):
        return type_(str(ret))
    raise TypeError(f"unsupported type at {key=}: {type(ret)=!r} ({ret=!r})")


class RegistrationGroup(NamedTuple):
    "group of images on which there is evaluation and/or prediction tasks"
    image_group: cm.ImageGroup
    eval_annotation_groups: Set[cm.AnnotationGroup]
    pred_annotations: Set[cm.Annotation]

    def is_empty(self) -> bool:
        return not self.eval_annotation_groups and not self.pred_annotations


class JobParameters(NamedTuple):
    "parsed parameters for the job"
    image_crop: ImageCrop
    image_ordering: ImageOrdering
    align_toward_reference: bool
    registration_type: RegistrationType
    compose_non_rigid: bool
    max_proc_size: int
    micro_max_proc_size: int

    groups: Sequence[RegistrationGroup]
    whitelist_ids: List[int]
    grayscale_images: List[int]

    full_log: bool

    @staticmethod
    def check(ns):
        "raise ValueError on bad parameters"

        full_log = get(ns, "full_log", eb, False)

        ## parse common parameters
        image_crop = get(ns, "image_crop", ImageCrop, ImageCrop.ALL)
        image_ordering = get(ns, "image_ordering", ImageOrdering, ImageOrdering.AUTO)
        align_toward_reference = get(ns, "align_toward_reference", eb, True)
        registration_type = get(
            ns, "registration_type", RegistrationType, RegistrationType.NON_RIGID
        )
        compose_non_rigid = get(ns, "compose_non_rigid", eb, False)
        max_proc_size = get(ns, "max_proc_size", ei)
        micro_max_proc_size = get(ns, "micro_max_proc_size", ei)

        if max_proc_size is None:
            max_proc_size = registration.DEFAULT_MAX_PROCESSED_IMG_SIZE
        if micro_max_proc_size is None:
            micro_max_proc_size = registration.DEFAULT_MAX_NON_RIGID_REG_SIZE
            # avoid wasting space for the non-micro registration if micro is not needed
            if registration_type != RegistrationType.MICRO:
                micro_max_proc_size = max_proc_size

        if max_proc_size <= 0:
            raise ValueError(f"{max_proc_size=} <= 0")

        if micro_max_proc_size < max_proc_size:
            raise ValueError(f"{micro_max_proc_size=} < {max_proc_size=}")

        ## parse Cytomine parameters
        eval_ag_ids = get(ns, "eval_annotation_groups", eil, eil(""))
        eval_ig_ids = get(ns, "eval_image_groups", eil, eil(""))
        pred_an_ids = get(ns, "data_annotations", eil, eil(""))
        whitelist_ids = get(ns, "image_whitelist", eil, eil(""))
        grayscale_ids = get(ns, "fix_grayscale_images", eil, eil(""))

        pred_all = get(ns, "pred_all_annotations", eb, False)
        if pred_all and pred_an_ids:
            warnings.warn("selected annotations in data_annotations are ignored")
            pred_an_ids = []

        # caches to avoid duplicate API calls
        all_image_group: Dict[int, cm.ImageGroup] = {}
        all_annotation_group: Dict[int, cm.AnnotationGroup] = {}
        ig_to_ag: DefaultDict[int, Set[cm.AnnotationGroup]] = defaultdict(set)
        ig_to_an: DefaultDict[int, Set[cm.Annotation]] = defaultdict(set)

        # fetch all annotation groups
        for ag_id in eval_ag_ids:
            if ag_id in all_annotation_group:
                continue
            ag = cm.AnnotationGroup().fetch(ag_id)
            if not ag:
                raise ValueError(f"could not fetch AnnotationGroup wih {ag_id=!r}")
            all_annotation_group[ag_id] = ag

            # get all image group before the registration
            ig_id: int = ag.imageGroup
            ig = _fetch_image_group(ig_id)
            if not ig:
                raise ValueError(f"could not fetch ImageGroup with {ig_id=!r}")

            all_image_group[ig.id] = ig
            ig_to_ag[ig_id].add(ag)

        # fetch all image groups
        for ig_id in eval_ig_ids:
            if ig_id in all_image_group:
                continue
            ig = _fetch_image_group(ig_id)
            all_image_group[ig_id] = ig

            # fetch all annotation group in the image group !
            agc = cm.AnnotationGroupCollection().fetch_with_filter("imagegroup", ig.id)
            if agc is False:  # API error
                raise ValueError("could not fetch AnnotationGroupCollection")

            if not agc:  # empty collection
                warnings.warn(f"ImageGroup {ig_id} has no AnnotationGroup")

            for ag in agc:
                if ag.id in all_annotation_group:
                    continue
                all_annotation_group[ag.id] = ag
                ig_to_ag[ig_id].add(ag)

        if pred_all:
            for ig_id, ig in all_image_group.items():
                images = _fetch_image_col(ig_id)
                if images is False:
                    raise ValueError(f"unable to fetch images of {ig_id=!r}")

                for img in images:
                    if whitelist_ids and img.id not in whitelist_ids:
                        continue
                    ac = cm.AnnotationCollection()
                    ac.image = img.id
                    ac.showWKT = True
                    ac.showTerm = True
                    ac = ac.fetch()

                    if ac is False:
                        raise ValueError(f"unable to fetch annotations for {img.id=!r}")

                    an: cm.Annotation
                    for an in ac:
                        ig_to_an[ig_id].add(an)

        # fetch all annotations
        for an_id in pred_an_ids:
            an = cm.Annotation().fetch(an_id)
            if an is False:
                raise ValueError(f"cannot fetch annotation with id={an_id}")

            if whitelist_ids and an.image not in whitelist_ids:
                raise ValueError(f"{an.image=} is not in {whitelist_ids=}")

            ig_ii_c = cm.ImageGroupImageInstanceCollection().fetch_with_filter(
                "imageinstance", an.image
            )
            if not ig_ii_c:
                raise ValueError(f"could not fetch IG_II_c for {an.image=!r}")

            ig_ii: cm.ImageGroupImageInstance
            for ig_ii in ig_ii_c:

                if ig_ii.group not in all_image_group:
                    image_group = _fetch_image_group(ig_ii.group)
                    if not image_group:
                        raise ValueError(f"could not fetch image group ({ig_ii.group})")
                    all_image_group[image_group.id] = image_group

                ig_to_an[ig_ii.group].add(an)

        all_image_ids: Set[int] = set()
        for ig_id in all_image_group:
            for img in _fetch_image_col(ig_id):
                all_image_ids.add(img.id)
        for img_id in whitelist_ids:
            if img_id not in all_image_ids:
                warnings.warn(f"whitelisted {img_id=} not given in any groups")
        for img_id in grayscale_ids:
            if img_id not in all_image_ids:
                warnings.warn(f"grayscale {img_id=} not given in any groups")

        groups = [
            RegistrationGroup(ig, ig_to_ag[ig_id], ig_to_an[ig_id])
            for ig_id, ig in all_image_group.items()
        ]

        return JobParameters(
            image_crop=image_crop,
            image_ordering=image_ordering,
            align_toward_reference=align_toward_reference,
            registration_type=registration_type,
            compose_non_rigid=compose_non_rigid,
            max_proc_size=max_proc_size,
            micro_max_proc_size=micro_max_proc_size,
            groups=groups,
            whitelist_ids=whitelist_ids,
            grayscale_images=grayscale_ids,
            full_log=full_log,
        )

    def __repr__(self) -> str:
        asdict = self._asdict()
        if self.groups:
            asdict["groups"] = [
                (
                    a.image_group.id,
                    [ag.id for ag in a.eval_annotation_groups],
                    [a.id for a in a.pred_annotations],
                )
                for a in self.groups
            ]

        return pretty_repr(asdict)
