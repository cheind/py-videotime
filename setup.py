
from distutils.core import setup

setup(
    name='py-videotime',
    version=open('videotime/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Extract time overlays in videos.',
    url='https://github.com/cheind/py-videotime',
    packages=['videotime', 'videotime.apps'],
)