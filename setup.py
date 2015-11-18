from distutils.core import setup

import sys

name = 'mldatasets'

package_dir = 'python2_noedit' if sys.version_info[0] < 3 else name

setup(
    name=name,
    version='0.0.1',
    packages=['.'],
    package_dir={'':package_dir},
    url='',
    license='MIT',
    author='yk',
    author_email='',
    description=''
)
