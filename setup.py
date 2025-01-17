# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/01 10:17
# @author   :Mo
# @function :setup of macro-correct


from setuptools import find_packages, setup
import codecs
import os

from macro_correct.version import __version__


# Package meta-data.
NAME = "macro-correct"
DESCRIPTION = "macro-correct"
URL = "https://github.com/yongzhuo/macro-correct"
EMAIL = "1903865025@qq.com"
AUTHOR = "yongzhuo"
LICENSE = "Apache"

with codecs.open("README.md", "r", "utf-8") as reader:
    long_description = reader.read()
with codecs.open("requirements.txt", "r", "utf-8") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))


setup(long_description_content_type="text/markdown",
      long_description=long_description,
      install_requires=install_requires,
      packages=find_packages(),
      description=DESCRIPTION,
      license=LICENSE,
      version=__version__,
      author_email=EMAIL,
      author=AUTHOR,
      name=NAME,
      url=URL,

      package_data={"macro_correct": [
          "*.*",
          "pytorch_sequencelabeling/*",
          "pytorch_textcorrection/*",
          "output/*"
          "task/*",
          "task/correct/*",
          "task/punct/*",
          "output/*.config",
          "output/*.json",
      ],
          # "": ["*.json"]
      },

      # data_files=[
      #     (".", ["macro_correct/output/confusion_dict.json",
      #            "macro_correct/output/csc.config",
      #            ]
      #      )
      # ],
      classifiers=["License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3.5",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11",
                   "Programming Language :: Python :: Implementation :: CPython",
                   "Programming Language :: Python :: Implementation :: PyPy"],
      )


if __name__ == "__main__":
    print("setup ok!")


# 打包与安装
# step:
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install

# or

# python setup.py bdist_wheel --universal
# twine upload dist/*

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=1.4

