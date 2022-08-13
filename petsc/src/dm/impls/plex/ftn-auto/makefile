
#requiresdefine   'PETSC_HAVE_FORTRAN'
ALL: lib
CPPFLAGS = ${NETCFD_INCLUDE} ${EXODUSII_INCLUDE}
CFLAGS   =
FFLAGS   =
SOURCEC  = plexindicesf.c plexnaturalf.c plexprojectf.c plexgeometryf.c plexreorderf.c plexpartitionf.c plexsubmeshf.c plexorientf.c plextreef.c plexgmshf.c plexexodusiif.c plexf.c plexcreatef.c plexcgnsf.c plexegadsf.c plexcheckinterfacef.c plexpointf.c plexdistributef.c plexfemf.c plexrefinef.c plexmetricf.c plexfvmf.c plexinterpolatef.c
SOURCEF  =
SOURCEH  = 
DIRS     =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/impls/plex/ftn-auto/
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
