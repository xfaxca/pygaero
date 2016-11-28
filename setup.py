#!C:/Python34 python

from distutils.core import setup

setup(name='pygaero',
      version='1.3',
      description='Python tools for FIGAERO HR-ToF-CIMS data processing and analysis.',
      author='Cameron Faxon',
      license='GNU GPLv3',
      author_email='Cameron@tutanota.com',
      url='https://github.com/gnomeone/pygaero',
      packages=['pygaero'],
      requires=['pandas', 'numpy', 'periodic', 'pathlib2', 'peakutils', 'sklearn', 'matplotlib'])
