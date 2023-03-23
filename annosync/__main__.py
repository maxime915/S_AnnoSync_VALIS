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
        home = pathlib.Path(os.environ.get("WORKDIR", ".")).resolve() / label
        home.mkdir(parents=True, exist_ok=False)

        if g_scratch := os.environ.get("GLOBALSCRATCH", None):
            global_scratch = pathlib.Path(g_scratch) / label
        else:
            global_scratch = home / "scratch"
        global_scratch.mkdir(parents=True, exist_ok=False)

        if l_scratch := os.environ.get("LOCALSCRATCH", None):
            local_scratch = pathlib.Path(l_scratch) / label
            local_scratch.mkdir(parents=True, exist_ok=False)
        else:
            local_scratch = global_scratch

        logger.debug("home = %s", str(home))
        logger.debug("local scratch = %s", str(local_scratch))
        logger.debug("global scratch = %s", str(global_scratch))

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
                cytomine_job=job,
                parameters=parameters,
                logger=logger,
            ).run()

        job.job.update(
            status=cm.Job.TERMINATED, progress=100, status_comment="Job terminated"
        )

main(sys.argv[1:])
