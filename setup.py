from setuptools import setup, find_packages
from os import path

setup(
      name='LDA-project-19',
      version=1.5,
      description='Latent Dirichlet Allocation',
      url='https://github.com/mingjiezhao/LDA_final_project_SP2019',
      author='Mingjie Zhao, Ziwei Zhu',
      author_email='mz136@duke.edu',
      classifiers=[
                  'Development Status :: 3 - Alpha',
                  'Intended Audience :: Developers',
                  'Topic :: Software Development :: Libraries :: Python Modules',
                  'License :: OSI Approved :: MIT License',
                  'Programming Language :: Python :: 3',
                  'Programming Language :: Python :: 3.4',
                  'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                  ],
      py_modules = ['LDA-project-19'],
      packages=find_packages(),

      python_requires='>=3',
      )