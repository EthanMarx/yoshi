import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable

import law
import luigi
from law.contrib import singularity

root = str(Path(__file__).resolve().parent.parent.parent)


class YoshiSandbox(singularity.SingularitySandbox):
    sandbox_type = "yoshi"

    @property
    def data_directories(self):
        return ["/cvmfs", "/hdfs", "/gpfs", "/ceph", "/hadoop"]

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[root] = "/opt/yoshi"

        # bind data directories if they
        # exist on this cluster
        for dir in self.data_directories:
            if os.path.exists(dir):
                volumes[dir] = dir
        return volumes


class YoshiTask(law.SandboxTask):
    image = luigi.Parameter()
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    container_root = luigi.Parameter(
        default=os.getenv("YOSHI_CONTAINER_ROOT", "")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(self.image):
            self.image = os.path.join(self.container_root, self.image)

        if not os.path.exists(self.image):
            raise ValueError(
                f"Could not find path to container image {self.image}"
            )

    @property
    def sandbox(self):
        return f"yoshi::{self.image}"

    def singularity_forward_law(self) -> bool:
        return False

    def sandbox_law_executable(self):
        # law executable that is used inside the sandbox
        return "/opt/env/bin/law"

    @property
    def singularity_allow_binds(self) -> bool:
        return True

    @property
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter

    def sandbox_env(self, _):
        if self.gpus:
            return {"CUDA_VISIBLE_DEVICES": self.gpus}
        return {}

    @property
    def python(self) -> str:
        return "/usr/local/bin/python"

    @property
    def command(self) -> str:
        return [self.command, "-c", "print('Hello world')"]

    def run(self):
        try:
            subprocess.run(
                self.command, capture_output=True, check=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Command '{}' failed with return code {} "
                "and stderr:\n{}".format(
                    shlex.join(e.cmd), e.returncode, e.stderr
                )
            ) from None
