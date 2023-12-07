import os
import re
import shutil
from pathlib import Path

from setuptools import Command, find_packages, setup

_deps = [
    "Pillow<10.0.0",
    "accelerate>=0.20.3",
    "av==9.2.0",  # Latest version of PyAV (10.0.0) has issues with audio stream.
    "beautifulsoup4",
    "black~=23.1",
    "codecarbon==1.2.0",
    "cookiecutter==1.7.3",
    "dataclasses",
    "datasets!=2.5.0",
    "decord==0.6.0",
    "deepspeed>=0.9.3",
    "diffusers",
    "dill<0.3.5",
    "evaluate>=0.2.0",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flax>=0.4.1,<=0.7.0",
    "fsspec<2023.10.0",
    "ftfy",
    "fugashi>=1.0",
    "GitPython<3.1.19",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.16.4,<1.0",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.4.1,<=0.4.13",
    "jaxlib>=0.4.1,<=0.4.13",
    "jieba",
    "kenlm",
    # Keras pin - this is to make sure Keras 3 doesn't destroy us. Remove or change when we have proper support.
    "keras<2.15",
    "keras-nlp>=0.3.1",
    "librosa",
    "nltk",
    "natten>=0.14.6",
    "numpy>=1.17",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "opencv-python",
    "optuna",
    "optax>=0.0.8,<=0.1.4",
    "packaging>=20.0",
    "parameterized",
    "phonemizer",
    "protobuf",
    "psutil",
    "pyyaml>=5.1",
    "pydantic<2",
    "pytest>=7.2.0",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "ray[tune]",
    "regex!=2019.12.17",
    "requests",
    "rhoknp>=1.1.0,<1.3.1",
    "rjieba",
    "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",
    "ruff>=0.0.241,<=0.0.259",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "safetensors>=0.3.1",
    "sagemaker>=2.31.0",
    "scikit-learn",
    "sentencepiece>=0.1.91,!=0.1.92",
    "sigopt",
    "starlette",
    "sudachipy>=0.6.6",
    "sudachidict_core>=20220729",
    "tensorboard",
    # TensorFlow pin. When changing this value, update examples/tensorflow/_tests_requirements.txt accordingly
    "tensorflow-cpu>=2.6,<2.15",
    "tensorflow>=2.6,<2.15",
    "tensorflow-text<2.15",
    "tf2onnx",
    "timeout-decorator",
    "timm",
    "tokenizers>=0.14,<0.15",
    "torch>=1.10,!=1.12.0",
    "torchaudio",
    "torchvision",
    "pyctcdecode>=0.4.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "urllib3<2.0.0",
    "uvicorn",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    deps["filelock"],  # filesystem locks, e.g., to prevent parallel downloads
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["pyyaml"],  # used for the model cards metadata
    deps["regex"],  # for OpenAI GPT
    deps["requests"],  # for downloading models over HTTPS
    deps["tqdm"],  # progress bars in model download and training scripts
    #    deps["huggingface-hub"],
    #    deps["tokenizers"],
    #    deps["safetensors"],
]

setup(
    name="torch-fire",
    include_package_data=True,
    zip_safe=False,
    install_requires=list(install_requires),
#    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)
