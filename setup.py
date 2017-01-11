#setup.py

from distutils.core import setup

setup(name='pygaero',
      version='1.50',
      description='Python tools for FIGAERO HR-ToF-CIMS data processing and analysis.',
      author='Cameron Faxon',
      license='GNU GPLv3',
      author_email='xfaxca@tutanota.com',
      url='https://github.com/xfaxca/pygaero',
      packages=['pygaero'],
      requires=['pandas', 'numpy', 'periodic', 'pathlib2', 'peakutils', 'sklearn', 'matplotlib'])
