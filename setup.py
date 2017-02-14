# setup.py

from setuptools import setup, find_packages

setup(name='pygaero',
      version='1.50',
      description='Python tools for FIGAERO HR-ToF-CIMS data processing and analysis.',
      author='Cameron Faxon',
      author_email='xfaxca@tutanota.com',
      license='GNU GPLv3',
      url='https://github.com/xfaxca/pygaero',
      package_data={'': ['elements.csv']},
      include_package_data=True,
      packages=find_packages(exclude=('logs', 'example')),
      install_requires=['matplotlib==1.5.1',
                         'numpy==1.11.2',
                         'pandas==0.18.1',
                         'pathlib2==2.1.0',
                         'PeakUtils==1.0.3',
                         'scikit_learn==0.18.1'])
