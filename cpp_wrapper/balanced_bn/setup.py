from glob import glob
from setuptools import setup
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

_ext_sources = glob("*.cpp") + glob("*.cu")

requirements = ["torch>=1.4"]

setup(name="balanced_bn",
    version="0.0.1",
    author="Yongyi Su",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="balanced_bn",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            # include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    )