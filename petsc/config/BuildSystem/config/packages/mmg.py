import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'e01905479683d27866306312424bb41fdc817f5e' # jolivet/feature-mmg-install-3.18.0 apr-10-2022
    self.download         = ['git://https://github.com/prj-/mmg.git','https://github.com/prj-/mmg/archive/'+self.gitcommit+'.tar.gz']
    self.versionname      = 'MMG_VERSION_RELEASE'
    self.includes         = ['mmg/libmmg.h']
    self.liblist          = [['libmmg.a','libmmg3d.a']]
    self.functions        = ['MMG5_paramUsage1']
    self.precisions       = ['double']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags = framework.require('config.compilerFlags',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.ptscotch      = framework.require('config.packages.PTScotch',self)
    self.deps          = [self.mathlib,self.ptscotch]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_ELAS=OFF')
    args.append('-DUSE_VTK=OFF')
    args.append('-DUSE_POINTMAP=ON')
    args.append('-DSCOTCH_DIR:STRING="'+self.ptscotch.directory+'"')
    return args
