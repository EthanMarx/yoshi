import os

import law
import luigi

from yoshi.tasks.base import YoshiTask
from yoshi.tasks.workflow import LDGCondorWorkflow

DATAFIND_ENV_VARS = [
    "KRB5_KTNAME",
    "X509_USER_PROXY",
    "GWDATAFIND_SERVER",
    "NDSSERVER",
    "LIGO_USERNAME",
]


class DataTask(YoshiTask):
    job_log = luigi.Parameter(default="")

    @property
    def job_type(self):
        return self.__class__.__name__.lower()

    @property
    def python(self):
        return "/opt/env/bin/python"

    @property
    def cli(self):
        cmd = [self.python, "/opt/yoshi/data/data"]
        if self.job_log:
            cmd += ["--log-file", self.job_log]
        return cmd + [self.job_type]

    def sandbox_env(self, env):
        env = super().sandbox_env(env)
        for envvar in DATAFIND_ENV_VARS:
            value = os.getenv(envvar)
            if value is not None:
                env[envvar] = value
        return env


class Query(DataTask):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    output_file = luigi.Parameter()
    min_duration = luigi.FloatParameter(default=0)
    flags = luigi.ListParameter(default=["DCS-ANALYSIS_READY_C01:1"])
    ifo = luigi.Parameter(default="H1")

    def output(self):
        return law.LocalFileTarget(self.output_file)

    @property
    def command(self):
        args = [
            "--start",
            str(self.start),
            "--end",
            str(self.end),
            "--output-file",
            self.output().path,
        ]
        for flag in self.flags:
            args.append("--flags+=" + self.ifo + ":" + flag)
        if self.min_duration > 0:
            args.append(f"--min_duration={self.min_duration}")
        return self.cli + args


class Fetch(DataTask, law.LocalWorkflow, LDGCondorWorkflow):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    min_duration = luigi.FloatParameter(default=0)
    max_duration = luigi.FloatParameter(default=-1)
    data_dir = luigi.Parameter()
    prefix = luigi.Parameter(default="yoshi")
    flags = luigi.ListParameter(default=["DCS-ANALYSIS_READY_C01:1"])
    segments_file = luigi.Parameter(default="")
    channels = luigi.ListParameter(default=["H1:GDS-CALIB_STRAIN"])
    log_dir = luigi.OptionalParameter()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        os.makedirs(self.data_dir, exist_ok=True)
        if not self.segments_file:
            self.segments_file = os.path.join(self.data_dir, "segments.txt")

        if self.job_log and not os.path.isabs(self.job_log):
            os.makedirs(self.log_dir, exist_ok=True)
            self.job_log = os.path.join(self.log_dir, self.job_log)

    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.input()["segments"].exists()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        segments = self.input()["segments"].load().splitlines()[1:]
        branch_map, i = {}, 1
        for row in segments:
            row = row.split("\t")
            start, duration = map(float, row[1::2])
            step = duration if self.max_duration == -1 else self.max_duration
            num_steps = (duration - 1) // step + 1

            for j in range(int(num_steps)):
                segstart = start + j * step
                segdur = min(start + duration - segstart, step)
                branch_map[i] = (segstart, segdur)
                i += 1
        return branch_map

    def workflow_requires(self):
        reqs = super().workflow_requires()

        kwargs = {}
        if self.job_log:
            log_file = law.LocalFileTarget(self.job_log)
            log_file = log_file.parent.child("query.log", type="f")
            kwargs["job_log"] = log_file.path
        reqs["segments"] = Query.req(
            self, output_file=self.segments_file, **kwargs
        )
        return reqs

    @workflow_condition.output
    def output(self):
        start, duration = self.branch_data
        start = int(float(start))
        duration = int(float(duration))
        fname = f"{self.prefix}-{start}-{duration}.hdf5"

        target = law.LocalDirectoryTarget(self.data_dir)
        target = target.child(fname, type="f")
        return target

    @property
    def command(self):
        start, duration = self.branch_data
        start = int(float(start))
        duration = int(float(duration))

        if self.job_log:
            log_file = law.LocalFileTarget(self.job_log)
            fname = log_file.basename[::-1].split(".", maxsplit=1)
            if len(fname) > 1:
                ext, fname = fname
                ext = "." + ext[::-1]
            else:
                ext = ""

            fname = fname[::-1]
            fname = fname + f"-{start}-{duration}{ext}"
            log_file = log_file.sibling(fname, type="f")
            self.job_log = log_file.path

        args = [
            "--start",
            str(start),
            "--end",
            str(start + duration),
            "--sample-rate",
            str(self.sample_rate),
            "--prefix",
            self.prefix,
            "--output-directory",
            self.data_dir,
            "--channels",
            "[" + ",".join(self.channels) + "]",
        ]
        return self.cli + args
