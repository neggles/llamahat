import os
import subprocess
import sys
import warnings
from pathlib import Path

import setuptools
import torch
from packaging.version import Version, parse
from torch import version as torch_version
from torch.utils import cpp_extension as cext
from torch.utils.cpp_extension import CUDA_HOME, IS_HIP_EXTENSION, IS_WINDOWS, BuildExtension, CUDAExtension

DEF_ARCH_LIST = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# Ninja needs absolute include paths
SCRIPT_DIR = Path(__file__).parent
C_SRC_DIR = SCRIPT_DIR / "csrc"
C_INCLUDE_DIRS = [
    str(C_SRC_DIR / "include"),
]


def get_cuda_bare_metal_version(cuda_dir) -> tuple[str, Version]:
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available? "
        + "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        + "only images whose names contain 'devel' will provide nvcc."
    )


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    warnings.warn(
        "\n"
        + "Torch did not find any available GPUs on this system.\n"
        + "If you are cross-compiling, this is expected, and can be safely ignored.\n"
        + "By default we will compile for Volta (CC 7.0), Turing (CC 7.5), Ampere (CC 8.0 to 8.6),\n"
        + "Ada (CC 8.9), and Hopper (CC 9.0), with PTX for forward-compatibility.\n"
        + "If you would like to compile for a specific architecture, please run\n"
        + 'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        + "\n"
        + "NOTE: While you *can* compile this for Pascal (CC 6.x), it's not supported due to a lack of\n"
        + "meaningful fp16 performance on all Pascal cards (save for P100). It might compile, and it\n"
        + "may even run, but it's unlikely to actually be usable.\n",
        UserWarning,
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = DEF_ARCH_LIST
        else:
            raise RuntimeError(f"llamahat requires CUDA 11.8 or newer, you have {bare_metal_version}")

# make sure ninja is available
cext.verify_ninja_availability()

extra_ldflags = list()
if cext.IS_WINDOWS:
    extra_ldflags.append("cublas.lib")
    if sys.base_prefix != sys.prefix:
        extra_ldflags.append(f"/LIBPATH:{Path(sys.base_prefix).joinpath('libs')}")

extra_cuda_cflags = ["-lineinfo"]
if IS_HIP_EXTENSION:
    extra_cuda_cflags.extend(["-U__HIP_NO_HALF_CONVERSIONS__", "-O3"])
else:
    extra_cuda_cflags.extend(["-O3"])


def get_cext_sources():
    workdir = Path.cwd().absolute()
    print(f"{__file__}: get_cext_sources() work dir: {Path.cwd().absolute()}")
    sources = [
        x.relative_to(workdir)
        for x in workdir.glob("csrc/**/*")
        if x.is_file() and x.suffix in [".cu", ".cpp", ".c"]
    ]
    if len(sources) == 0:
        raise RuntimeError(f"Could not find any source files in {workdir}/csrc")
    return [str(x) for x in sources]


setuptools.setup(
    ext_modules=[
        CUDAExtension(
            name="llamahat.cuda",
            sources=get_cext_sources(),
            include_dirs=C_INCLUDE_DIRS,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": extra_cuda_cflags,
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
