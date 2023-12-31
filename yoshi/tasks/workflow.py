import os

import law
import luigi
from law.contrib import htcondor

DATAFIND_ENV_VARS = [
    "KRB5_KTNAME",
    "X509_USER_PROXY",
    "GWDATAFIND_SERVER",
    "NDSSERVER",
    "LIGO_USERNAME",
]


class LDGCondorWorkflow(htcondor.HTCondorWorkflow):
    condor_directory = luigi.Parameter()
    accounting_group_user = luigi.Parameter(default=os.getenv("LIGO_USER"))
    accounting_group = luigi.Parameter(default=os.getenv("LIGO_GROUP"))
    request_disk = luigi.Parameter(default="1 GB")
    request_memory = luigi.Parameter(default="1 GB")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.htcondor_log_dir.touch()
        self.htcondor_output_directory().touch()

        # update location of where htcondor
        # job files are stored
        # TODO: law PR that makes this configuration
        # easier / more pythonic
        law.config.update(
            {
                "job": {
                    "job_file_dir": self.job_file_dir,
                    "job_file_dir_cleanup": "false",
                    "job_file_dir_mkdtemp": "false",
                }
            }
        )

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def htcondor_log_dir(self):
        return law.LocalDirectoryTarget(
            os.path.join(self.condor_directory, "logs")
        )

    @property
    def job_file_dir(self):
        return self.htcondor_output_directory().child("jobs", type="d").path

    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.condor_directory)

    def htcondor_use_local_scheduler(self):
        return True

    def htcondor_job_config(self, config, job_num, branches):
        environment = '"'
        for envvar in DATAFIND_ENV_VARS:
            environment += f"{envvar}={os.getenv(envvar)} "
        environment += f'PATH={os.getenv("PATH")}"'

        config.custom_content.append(("environment", environment))
        config.custom_content.append(("request_memory", self.request_memory))
        config.custom_content.append(("request_disk", self.request_disk))
        config.custom_content.append(
            ("accounting_group", self.accounting_group)
        )
        config.custom_content.append(
            ("accounting_group_user", self.accounting_group_user)
        )

        config.custom_content.append(
            (
                "log",
                os.path.join(self.log_dir, f"{self.name}-$(Cluster).log"),
            )
        )
        config.custom_content.append(
            (
                "output",
                os.path.join(self.log_dir, f"{self.name}-$(Cluster).out"),
            )
        )
        config.custom_content.append(
            (
                "error",
                os.path.join(self.log_dir, f"{self.name}-$(Cluster).err"),
            )
        )
        return config
