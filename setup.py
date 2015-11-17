from distutils.core import setup
import sys

package_dir = 'python2_noedit' if sys.version_info[0] < 3 else 'mldatasets'

setup(
    name='mldatasets',
    version='0.0.1',
    packages=['.'],
    package_dir={'': package_dir},
    url='',
    license='MIT',
    author='yk',
    author_email='',
    description=''
)
