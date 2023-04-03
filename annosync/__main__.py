import contextlib
import logging
import os
import pathlib
import sys

import cytomine
import cytomine.models as cm
from valis import registration

from .job_parameter import JobParameters
from .valis_job import VALISJob


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

"""
# TODO[cache-mp]

The idea was to use a R-W image cache such that all jobs could avoid downloading
multiple times the same images.

The issue faced, is that running multiple jobs in parallel leads to a race
condition which is very hard to remove (keeping the RW cache). Thus, the cache
is temporarily made read-only but this has a few limitations :
    - only original file format is allowed (otherwise the cache is too
        complicated to manage by hand) -> so no SLDC
    - the grayscale correction is not allowed (would corrupt the cache)

This text will be removed when a solution is implemented.
"""

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

        label = f"{job.software.name}-{job.job.id}"
        home = pathlib.Path(os.environ.get("WORKDIR", ".")).resolve()
        if not home.exists() and home.parent.exists():
            # use this as the workdir: no subfolder
            home.mkdir(parents=False, exist_ok=False)
        elif not home.exists() and not home.parent.exists():
            # possibly unintended
            raise ValueError(f"will not create {home!r}: parent dir not found")
        else:  # home.exists() is True
            home = home / label
            home.mkdir(parents=False, exist_ok=False)

        cache_label = f"{job.software.name}-{job.software.id}-cache-dir"
        if g_scratch := os.environ.get("GLOBALSCRATCH", None):
            global_scratch = pathlib.Path(g_scratch) / label
            image_cache = pathlib.Path(g_scratch) / cache_label
        else:
            global_scratch = home / "scratch"
            image_cache = global_scratch / "cache-dir"
        global_scratch.mkdir(parents=False, exist_ok=False)
        image_cache.mkdir(parents=False, exist_ok=True)  # should be reused

        if l_scratch := os.environ.get("LOCALSCRATCH", None):
            local_scratch = pathlib.Path(l_scratch) / label
            local_scratch.mkdir(parents=False, exist_ok=False)
        else:
            local_scratch = global_scratch

        logger.debug("home = %s", str(home))
        logger.debug("local scratch = %s", str(local_scratch))
        logger.debug("global scratch = %s", str(global_scratch))
        logger.debug("image cache = %s", str(image_cache))

        # check all parameters and fetch from Cytomine
        parameters = JobParameters.check(job.parameters)

        if not parameters.groups:
            raise ValueError("cannot operate on empty data")
        if any(rg.is_empty() for rg in parameters.groups):
            raise ValueError("at least one group is empty")

        with contextlib.ExitStack() as e:
            registration.init_jvm()
            e.callback(registration.kill_jvm)
            VALISJob(
                home_dir=home,
                local_scratch=local_scratch,
                global_scratch=global_scratch,
                image_cache=image_cache,
                cytomine_job=job,
                parameters=parameters,
                logger=logger,
            ).run()

        job.job.update(
            status=cm.Job.TERMINATED, progress=100, status_comment="Job terminated"
        )


main(sys.argv[1:])
