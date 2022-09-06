#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-petsc4py=1',
    '--with-cc=gcc',
    '--with-cxx=g++',
    '--download-fblaslapack=1',
    #'--download-openmpi=1',
    '--with-petsc4py',
    #'--with-precision=single',
    #'--download-mpich=1',
    '--with-mpi=0',
    '--with-cuda=1',
    '--with-debugging=0',
    '--COPTFLAGS= -g -O3 -march=native',
    '--CXXOPTFLAGS= -g -O3 -march=native', 
    '--FOPTFLAGS= -g -O3 -march=native',
    '--CUDAOPTFLAGS= -O3', 
    '--download-slepc',
    '--download-slepc-configure-arguments= --with-slepc4py'

  ]
  print(configure_options)
  configure.petsc_configure(configure_options)
