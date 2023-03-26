import logging
import pathlib
import shutil
from typing import Callable, List, NamedTuple, Tuple

import cytomine
import cytomine.models as cm
import numpy as np
import pandas as pd
import shapely
import shapely.errors
import shapely.wkt
import sldc_cytomine
import sldc_cytomine.dump
from shapely.affinity import affine_transform
from shapely.ops import transform
from shapely.validation import make_valid
from valis import registration

from .images import (
    CytominePIMSTile,
    dice_annotations,
    distance_annotations,
    fix_grayscale,
    image_shape,
    iou_annotations,
    tre_annotations,
)
from .job_parameter import (
    ImageOrdering,
    JobParameters,
    RegistrationGroup,
    RegistrationType,
    DownloadFormat,
)
from .utils import _fetch_image, _fetch_image_col, no_output


class VALISJob(NamedTuple):
    "VALISJob: an agglomeration of data useful for this job."

    """directory allocated to this job for important/small outputs of the program,
    configurations, ..."""
    home_dir: pathlib.Path

    """temporary directory : it can be erased by the cluster right after this job
    finishes. Any important file should be moved/copied from this partition to
    the scratch or home (depending on its size) before the end of the program."""
    local_scratch: pathlib.Path

    """other inputs and outputs : there is no backup guarantee for this directory,
    but it can store files reliably for a short duration (enough to inspect the
    results after a failure, ...). Perfect for I/O images, weights, ..."""
    global_scratch: pathlib.Path

    cytomine_job: cytomine.CytomineJob
    parameters: JobParameters
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
        image_col = _fetch_image_col(group.id)
        if not image_col:
            raise ValueError(f"cannot fetch all images for {group.id=}")

        images: List[cm.ImageInstance] = [
            img for img in image_col if self.allow(img.id)
        ]
        if not images:
            raise ValueError(f"filtering made {group.id=} empty")
        return images

    def thumb_path(
        self, group: RegistrationGroup, image: cm.ImageInstance
    ) -> pathlib.Path:
        base = self.get_slide_dir(group) / self.get_fname(image)
        if self.parameters.download_format == DownloadFormat.PNG:
            return base.with_suffix(".png")
        return base

    def download_images(self, group: RegistrationGroup):
        "download all images of the image group"

        if self.parameters.download_format == DownloadFormat.ORIGINAL:
            self._download_images_original(group)
        elif self.parameters.download_format == DownloadFormat.PNG:
            self._download_images_sldc(group)
        else:
            assert False, "incomplete switch"

    def _download_images_original(self, group: RegistrationGroup):
        
        def _dwl(img: cm.ImageInstance):
            img_path = self.thumb_path(group, img)
            result = img.download(str(img_path), override=True)
            if not result:
                raise ValueError(f"could not download image {img.id=!r} to {img_path=!r}")
            return img_path

        images = self.get_images(group.image_group)

        for img in images:
            try:
                img_path = _dwl(img)
                if img.id in self.parameters.grayscale_images:
                    fix_grayscale(img_path)
            except Exception as e:
                raise ValueError(
                    f"unable to download all images from {group.image_group.id=!r}"
                    ". See Cytomine's log for more information."
                ) from e


    def _download_images_sldc(self, group: RegistrationGroup):

        images = self.get_images(group.image_group)
        max_size = max(
            self.parameters.max_proc_size, self.parameters.micro_max_proc_size
        )

        img: cm.ImageInstance
        for img in images:
            img_path = self.thumb_path(group, img)

            img_max_size = max(img.width, img.height)
            zoom_level = int(max(0, np.floor(np.log2(img_max_size / max_size))))
            target = img_max_size // 2**zoom_level

            working_path = self.local_scratch / f"tmp-image-{img.id}"
            working_path.mkdir(parents=True, exist_ok=False)

            try:
                sldc_cytomine.dump.dump_region(
                    zone=img,
                    dest_pattern=str(img_path),
                    slide_class=sldc_cytomine.CytomineSlide,
                    tile_class=CytominePIMSTile,
                    zoom_level=zoom_level,
                    n_jobs=0,
                    working_path=working_path,
                )
            except Exception as e:
                raise ValueError(
                    f"could not download image {img.path} ({img.id}) "
                    f"for image group {group.image_group.name} "
                    f"({group.image_group.id})"
                    f"\n\t{img_path=!r}"
                    f"\n\t{zoom_level=!r}"
                    f"\n\t{working_path=!r}"
                ) from e
            finally:
                shutil.rmtree(working_path)

            try:
                actual = max(image_shape(img_path))
                if actual != target:
                    self.logger.error("id: %s, path: %s", img.id, img_path)
                    self.logger.error(
                        "Cytomine image shape: %s", (img.width, img.height)
                    )
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
        valis_args = {
            "src_dir": str(self.get_slide_dir(group)),
            "dst_dir": str(self.get_dst_dir(group)),
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

    def get_csv_dir(self, group: RegistrationGroup) -> pathlib.Path:
        "a directory where .CSV for the results can be stored"

        if not self.home_dir.exists():
            raise RuntimeError(f"{self.home_dir=!r} does not exist")
        csv_dir = self.home_dir / str(group.image_group.id) / "csv"
        csv_dir.mkdir(
            parents=True, exist_ok=True
        )  # could be called before get_slide_dir
        return csv_dir

    def get_slide_dir(self, group: RegistrationGroup) -> pathlib.Path:
        "a directory where images can be stored"

        if not self.global_scratch.exists():
            raise RuntimeError(f"{self.global_scratch=!r} does not exist")
        slide_dir = self.global_scratch / str(group.image_group.id) / "slides"
        slide_dir.mkdir(parents=True, exist_ok=True)
        return slide_dir

    def get_dst_dir(self, group: RegistrationGroup) -> pathlib.Path:
        "a directory where VALIS can store some data"

        if not self.home_dir.exists():
            raise RuntimeError(f"{self.home_dir=!r} does not exist")
        dst_dir = self.home_dir / str(group.image_group.id) / "valis-dst"
        dst_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir

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

        metrics: List[
            Tuple[
                str,
                Callable[[cm.Annotation, cm.Annotation], float],
                List[Tuple[int, int, int, int, int, float]],
            ]
        ] = [
            ("TRE", tre_annotations, []),
            ("IoU", iou_annotations, []),
            ("DIST", distance_annotations, []),
            ("Dice", dice_annotations, []),
        ]

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
            path = str(self.get_csv_dir(group) / filename)
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
