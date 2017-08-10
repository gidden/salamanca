#!/usr/bin/env python
from __future__ import print_function

import shutil

from setuptools import setup, Command, find_packages
from setuptools.command.install import install

# Thanks to http://patorjk.com/software/taag/
logo = r"""
     _______.     ___       __          ___      .___  ___.      ___      .__   __.   ______     ___      
    /       |    /   \     |  |        /   \     |   \/   |     /   \     |  \ |  |  /      |   /   \     
   |   (----`   /  ^  \    |  |       /  ^  \    |  \  /  |    /  ^  \    |   \|  | |  ,----'  /  ^  \    
    \   \      /  /_\  \   |  |      /  /_\  \   |  |\/|  |   /  /_\  \   |  . `  | |  |      /  /_\  \   
.----)   |    /  _____  \  |  `----./  _____  \  |  |  |  |  /  _____  \  |  |\   | |  `----./  _____  \  
|_______/    /__/     \__\ |_______/__/     \__\ |__|  |__| /__/     \__\ |__| \__|  \______/__/     \__\ 
                                                                                                          
"""

INFO = {
    'version': '0.1.0',
}


class Cmd(install):
    """Custom clean command to tidy up the project root."""

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        dirs = [
            'salamanca.egg-info',
            'build',
        ]
        for d in dirs:
            print('removing {}'.format(d))
            shutil.rmtree(d)


def main():
    print(logo)

    packages = find_packages()
    pack_dir = {
        'salamanca': 'salamanca',
    }
    entry_points = {
        'console_scripts': [
            'sal=salamanca.cli:main',
        ],
    }
    cmdclass = {
        'install': Cmd,
    }
    setup_kwargs = {
        "name": "salamanca",
        "version": INFO['version'],
        "description": 'Provides access to and operations on commonly used socio-economic indicators'
        'Trajectories',
        "author": 'Matthew Gidden',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'http://github.com/gidden/salamanca',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "cmdclass": cmdclass,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
