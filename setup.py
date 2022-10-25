#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from setuptools import (
    find_packages,
    setup,
)

import git_version


def version():
    try:
        with open("VERSION") as f:
            return f.readline().strip()
    except IOError:
        return "0.0.0"


def main():
    root_dir = os.path.dirname(__file__)

    with open(os.path.join(root_dir, "requirements.txt")) as f:
        requirements = [r.strip() for r in f]
        setup(
            name="neptune-tensorboard",
            version=version(),
            url="https://neptune.ai/",
            project_urls={
                "Tracker": "https://github.com/neptune-ai/neptune-tensorboard/issues",
                "Source": "https://github.com/neptune-ai/neptune-tensorboard",
                "Documentation": "https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking"
                "/tensorboard",
            },
            license="Apache License 2.0",
            author="neptune.ai",
            author_email="contact@neptune.ai",
            description="Neptune Tensorboard",
            long_description=__doc__,
            packages=find_packages(where="src"),
            platforms="any",
            install_requires=requirements,
            entry_points={"neptune.plugins": "tensorboard = neptune_tensorboard_plugin:sync"},
            cmdclass={
                "git_version": git_version.GitVersion,
            },
            classifiers=[
                # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
                "Development Status :: 4 - Beta",
                # 'Development Status :: 5 - Production/Stable',  # Switch to Stable when applicable
                "Environment :: Console",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: Apache Software License",
                "Natural Language :: English",
                "Operating System :: MacOS",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: POSIX",
                "Operating System :: Unix",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Topic :: Software Development :: Libraries :: Python Modules",
                "Programming Language :: Python :: Implementation :: CPython",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
            ],
            package_dir={"": "src"},
        )


if __name__ == "__main__":
    main()
