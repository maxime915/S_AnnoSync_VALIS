"""S_AnnoSync_VALIS: Annotation Synchronisation using VALIS

A Cytomine App to make annotation available in all images of an image group,
and compute an evaluation on some hand-drawn ground truths.
"""

import contextlib
import enum
import logging
import os
import pathlib
import sys
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import cv2
import cytomine
import cytomine.models as cm
import numpy as np
import pandas as pd
import shapely
import shapely.errors
import shapely.wkt
from shapely.affinity import affine_transform
from shapely.geometry.point import Point
from shapely.ops import transform
from shapely.validation import make_valid
from valis import registration

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


def image_shape(path: str) -> Tuple[int, int]:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"{path=!r} does not exist")

    return img.shape[1::-1]


def fix_grayscale(path: str):
    "save an RGB grayscale image as single channel"

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
    pathlib.Path(path).unlink()
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


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


class VALISJob(NamedTuple):

    cytomine_job: cytomine.CytomineJob
    parameters: JobParameters
    name: str
    base_dir: pathlib.Path = pathlib.Path(".")
    logger: logging.Logger = logging.getLogger("VALISJob")

    @property
    def silent(self):
        return not self.parameters.full_log

    def allow(self, image: int) -> bool:
        if not self.parameters.whitelist_ids:
            return True
        return image in self.parameters.whitelist_ids

    def update(self, progress: float, status: str):
        self.cytomine_job.job.update(
            status=cm.Job.RUNNING, progress=round(progress), statusComment=status
        )

    def get_images(self, group: cm.ImageGroup):
        images = _fetch_image_col(group.id)
        if not images:
            raise ValueError(f"cannot fetch all images for {group.id=}")

        images: List[cm.ImageInstance] = [img for img in images if self.allow(img.id)]
        if not images:
            raise ValueError(f"filtering made {group.id=} empty")
        return images

    def thumb_path(self, group: RegistrationGroup, image: cm.ImageInstance):
        base = self.get_slide_dir(group) / self.get_fname(image)
        return str(base.with_suffix(".png"))

    def download_images(self, group: RegistrationGroup):

        images = self.get_images(group.image_group)
        max_size = max(
            self.parameters.max_proc_size, self.parameters.micro_max_proc_size
        )

        img: cm.ImageInstance
        for img in images:
            try:
                img_path = self.thumb_path(group, img)

                target = min(max_size, max(img.width, img.height))
                img.window(0, 0, img.width, img.height, img_path, override=True, max_size=target)

                actual = max(image_shape(img_path))
                if actual != target:
                    self.logger.error("id: %s, path: %s", img.id, img_path)
                    self.logger.error("Cytomine image shape: %s", (img.width, img.height))
                    self.logger.error("requested max_size: %s", max_size)
                    self.logger.error("downloaded shape: %s", image_shape(img_path))
                    raise ValueError("downloaded image doesn't have the right size")

                # make grayscale image single channel
                if img.id in self.parameters.grayscale_images:
                    fix_grayscale(img_path)

            except ValueError as e:
                raise ValueError(
                    f"could not download image {img.path} ({img.id}) "
                    f"for image group {group.image_group.name} "
                    f"({group.image_group.id})"
                ) from e

    def get_valis_args(self, group: RegistrationGroup):
        slide_dir = self.get_slide_dir(group)
        valis_args = {
            "src_dir": str(slide_dir),
            "dst_dir": str(slide_dir.with_name("dst")),
            "name": self.name,
            "imgs_ordered": self.parameters.image_ordering != ImageOrdering.AUTO,
            "compose_non_rigid": self.parameters.compose_non_rigid,
            "align_to_reference": not self.parameters.align_toward_reference,
            "crop": self.parameters.image_crop.value,
            "max_image_dim_px": self.parameters.max_proc_size,
            "max_processed_image_dim_px": self.parameters.max_proc_size,
            "max_non_rigid_registartion_dim_px": self.parameters.max_proc_size,
        }

        # skip non rigid registrations
        if self.parameters.registration_type == RegistrationType.RIGID:
            valis_args["non_rigid_registrar_cls"] = None

        return valis_args

    def register(self, group: RegistrationGroup) -> registration.Valis:
        registrar = registration.Valis(**self.get_valis_args(group))

        # rigid and non-rigid registration
        with no_output(self.silent):
            rigid_registrar, _, _ = registrar.register()

        assert rigid_registrar is not None

        self.logger.info("non-micro registration done")
        self.logger.info("ref image: %s", registrar.reference_img_f)

        if self.parameters.registration_type == RegistrationType.MICRO:
            kwargs = {}
            kwargs[
                "max_non_rigid_registartion_dim_px"
            ] = self.parameters.micro_max_proc_size
            with no_output(self.silent):
                registrar.register_micro(**kwargs)

            self.logger.info("micro registration done")

        return registrar

    def get_slide_dir(self, group: RegistrationGroup) -> pathlib.Path:
        return self.base_dir / str(group.image_group.id) / "slides"

    def get_fname(self, image: cm.ImageInstance) -> str:
        if self.parameters.image_ordering == ImageOrdering.CREATED:
            return f"{image.created}_{image.filename}"
        return image.filename

    def get_image_slide(
        self,
        image: cm.ImageInstance,
        registrar: registration.Valis,
    ) -> registration.Slide:
        return registrar.get_slide(self.get_fname(image))

    def warp_annotation(
        self,
        annotation: cm.Annotation,
        group: RegistrationGroup,
        src_image: cm.ImageInstance,
        dst_image: cm.ImageInstance,
        registrar: registration.Valis,
    ):
        src_slide = self.get_image_slide(src_image, registrar)
        dst_slide = self.get_image_slide(dst_image, registrar)

        src_geometry_bl = shapely.wkt.loads(annotation.location)
        src_geometry_tl = affine_transform(
            src_geometry_bl, [1, 0, 0, -1, 0, src_image.height]
        )

        src_shape = image_shape(self.thumb_path(group, src_image))

        src_geometry_file_tl = affine_transform(
            src_geometry_tl,
            [
                src_shape[0] / src_image.width,
                0,
                0,
                src_shape[1] / src_image.height,
                0,
                0,
            ],
        )

        def warper_(x, y, z=None):
            assert z is None
            xy = np.stack([x, y], axis=1)
            warped_xy = src_slide.warp_xy_from_to(xy, dst_slide)

            return warped_xy[:, 0], warped_xy[:, 1]

        dst_geometry_file_tl = transform(warper_, src_geometry_file_tl)

        dst_shape = image_shape(self.thumb_path(group, dst_image))

        dst_geometry_tl = affine_transform(
            dst_geometry_file_tl,
            [
                dst_image.width / dst_shape[0],
                0,
                0,
                dst_image.height / dst_shape[1],
                0,
                0,
            ],
        )
        dst_geometry_bl = affine_transform(
            dst_geometry_tl, [1, 0, 0, -1, 0, dst_image.height]
        )

        return cm.Annotation(
            shapely.wkt.dumps(make_valid(dst_geometry_bl)),
            dst_image.id,
            annotation.term,
            annotation.project,
        )

    def evaluate(self, group: RegistrationGroup, registrar: registration.Valis):
        if not group.eval_annotation_groups:
            self.logger.info(
                "no annotation group to evaluate on, for imagegroup=%d",
                group.image_group.id,
            )
            return

        self.logger.info(
            "evaluation on %d annotation groups", len(group.eval_annotation_groups)
        )

        metrics = [
            ("TRE", tre_annotations, []),
            ("IoU", iou_annotations, []),
            ("DIST", distance_annotations, []),
            ("Dice", dice_annotations, []),
        ]  # type: list[tuple[str, Callable[[cm.Annotation, cm.Annotation], float], list[float]]]

        for an_group in group.eval_annotation_groups:
            an_coll = cm.AnnotationCollection(group=an_group.id)
            an_coll.project = group.image_group.project
            an_coll.showWKT = True
            an_coll.showTerm = True
            if an_coll.fetch() is False:
                raise ValueError(
                    f"unable to fetch annotation collection for "
                    f"{an_group.id=} and {group.image_group.project=}"
                )

            self.logger.info(
                "annotation group (id=%d) contains %d annotations",
                an_group.id,
                len(an_coll),
            )

            an: cm.Annotation
            for an in an_coll:
                if not self.allow(an.image):
                    continue

                src_img = _fetch_image(an.image)
                if not src_img:
                    raise ValueError(f"cannot fetch ImageInstance with id={an.image}")

                an_gt: cm.Annotation
                for an_gt in an_coll:
                    if an_gt.image == an.image:
                        continue

                    if not self.allow(an_gt.image):
                        continue

                    dst_img = _fetch_image(an_gt.image)
                    if not dst_img:
                        raise ValueError(
                            f"cannot fetch ImageInstance with id={an_gt.image}"
                        )

                    # warp an to the image of an_gt
                    pred = self.warp_annotation(an, group, src_img, dst_img, registrar)
                    base_row = (an_group.id, an.id, src_img.id, an_gt.id, dst_img.id)

                    for label, metric, rows in metrics:
                        try:
                            value = metric(pred, an_gt)
                        except TypeError:
                            continue  # just skip that metric
                        except shapely.errors.ShapelyError as e:
                            self.logger.error(
                                "unable to compute %s between an=%d (img=%d) and "
                                "an_gt=%d (img_gt=%d) "
                                "valid an: %s, valid pred: %s, valid gt: %s",
                                label,
                                an.id,
                                src_img.id,
                                an_gt.id,
                                dst_img.id,
                                shapely.wkt.loads(an.location).is_valid,
                                shapely.wkt.loads(pred.location).is_valid,
                                shapely.wkt.loads(an_gt.location).is_valid,
                            )
                            self.logger.exception(e)
                            continue

                        rows.append(base_row + (value,))

        for label, _, values in metrics:
            if not values:
                continue
            key = label.lower()
            filename = group.image_group.name + f"-{group.image_group.id}-{key}.csv"
            path = str(self.base_dir / filename)
            pd.DataFrame(
                values,
                columns=[
                    "annotation_group",
                    "annotation_src",
                    "image_src",
                    "annotation_gt",
                    "image_gt",
                    key,
                ],
            ).to_csv(path)
            job_data = cm.JobData(
                id_job=self.cytomine_job.job.id,
                key=f"{label} ({group.image_group.id})",
                filename=filename,
            )
            job_data = job_data.save()
            job_data.upload(path)

    def predict(self, group: RegistrationGroup, registrar: registration.Valis):

        images = self.get_images(group.image_group)

        self.logger.info(
            "making prediction on %d annotations", len(group.pred_annotations)
        )
        self.logger.info("these annotations will be mapped to %d images", len(images))
        for an in group.pred_annotations:
            # get source slide
            src_img = _fetch_image(an.image)
            if not src_img:
                raise ValueError(f"cannot fetch ImageInstance with id={an.image}")

            annotation_collection = cm.AnnotationCollection()
            ag = cm.AnnotationGroup(src_img.project, group.image_group.id)
            if ag.save() is False:
                raise ValueError("cannot create annotation group")

            img: cm.ImageInstance
            for img in images:
                # warp annotation from to
                warped_an = self.warp_annotation(an, group, src_img, img, registrar)

                # upload
                if (warped_an := warped_an.save()) is False:
                    raise ValueError("could not save new annotation")
                annotation_collection.append(warped_an)

            for an in annotation_collection:
                al = cm.AnnotationLink(id_annotation=an.id, id_annotation_group=ag.id)
                if al.save() is False:
                    raise ValueError("could not create annotation link")

    def run(self):
        def prog_it(progress: float, idx: int):
            "progress: float in [0., 100.0)\nidx: int in [0, len(groups))"
            return (100.0 * float(idx) + progress) / len(self.parameters.groups)

        self.logger.info("starting on %d groups", len(self.parameters.groups))

        self.logger.info("parsed parameters: %s", self.parameters)

        for idx, group in enumerate(self.parameters.groups):
            self.update(prog_it(0.1, idx), f"starting on {group.image_group.id=}")

            # get images and perform registration
            self.update(prog_it(0.2, idx), "downloading images")
            self.download_images(group)
            self.update(prog_it(39.9, idx), "done: downloading all images")

            self.update(prog_it(40.0, idx), "registering all images")
            registrar = self.register(group)
            self.update(prog_it(79.9, idx), "done: registering all images")

            # evaluation on ground truths data
            self.update(prog_it(80.0, idx), "evaluating performances")
            self.evaluate(group, registrar)
            self.update(prog_it(89.9, idx), "done: evaluating performances")

            # make some predictions
            self.update(prog_it(90.0, idx), "making predictions")
            self.predict(group, registrar)
            self.update(prog_it(99.9, idx), "done: making predictions")

        self.update(100.0, "done")


def _get_log_formatter():
    return logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)s] [%(levelname)s] : %(message)s",
        datefmt="%j %H:%M:%S",
    )


def _logger_filter(record: logging.LogRecord) -> bool:
    if record.name == "root":
        return True

    if record.name == "cytomine.client":
        record.name = "cyt-client"
        return record.levelno != logging.DEBUG

    return False


def main(arguments):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_logger_filter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.NOTSET)
    stream_handler.setFormatter(_get_log_formatter())
    stream_handler.addFilter(_logger_filter)
    logger.addHandler(stream_handler)

    with cytomine.CytomineJob.from_cli(arguments) as job:

        job.job.update(
            status=cm.Job.RUNNING, progress=0, status_comment="Initialization"
        )

        base_dir = pathlib.Path(f"./valis-slides-{job.job.id}")
        base_dir.mkdir(exist_ok=True, parents=False)

        # check all parameters and fetch from Cytomine
        parameters = JobParameters.check(job.parameters)

        if not parameters.groups:
            raise ValueError("cannot operate on empty data")
        if any(rg.is_empty() for rg in parameters.groups):
            raise ValueError("at least one group is empty")

        with contextlib.ExitStack() as e:
            registration.init_jvm()
            e.callback(registration.kill_jvm)
            VALISJob(job, parameters, "main", base_dir, logger).run()

        job.job.update(
            status=cm.Job.TERMINATED, progress=100, status_comment="Job terminated"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
