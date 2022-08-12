
#include <petsc/private/pcmgimpl.h>       /*I "petscksp.h" I*/

/* ---------------------------------------------------------------------------*/
/*@C
   PCMGResidualDefault - Default routine to calculate the residual.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution

   Output Parameter:
.  r - location to store the residual

   Level: developer

.seealso: `PCMGSetResidual()`
@*/
PetscErrorCode  PCMGResidualDefault(Mat mat,Vec b,Vec x,Vec r)
{
  PetscFunctionBegin;
  PetscCall(MatResidual(mat,b,x,r));
  PetscFunctionReturn(0);
}

/*@C
   PCMGResidualTransposeDefault - Default routine to calculate the residual of the transposed linear system

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution

   Output Parameter:
.  r - location to store the residual

   Level: developer

.seealso: `PCMGSetResidualTranspose()`
@*/
PetscErrorCode PCMGResidualTransposeDefault(Mat mat,Vec b,Vec x,Vec r)
{
  PetscFunctionBegin;
  PetscCall(MatMultTranspose(mat,x,r));
  PetscCall(VecAYPX(r,-1.0,b));
  PetscFunctionReturn(0);
}

/*@C
   PCMGMatResidualDefault - Default routine to calculate the residual.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution

   Output Parameter:
.  r - location to store the residual

   Level: developer

.seealso: `PCMGSetMatResidual()`
@*/
PetscErrorCode  PCMGMatResidualDefault(Mat mat,Mat b,Mat x,Mat r)
{
  PetscFunctionBegin;
  PetscCall(MatMatMult(mat,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&r));
  PetscCall(MatAYPX(r,-1.0,b,UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/*@C
   PCMGMatResidualTransposeDefault - Default routine to calculate the residual of the transposed linear system

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution

   Output Parameter:
.  r - location to store the residual

   Level: developer

.seealso: `PCMGSetMatResidualTranspose()`
@*/
PetscErrorCode PCMGMatResidualTransposeDefault(Mat mat,Mat b,Mat x,Mat r)
{
  PetscFunctionBegin;
  PetscCall(MatTransposeMatMult(mat,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&r));
  PetscCall(MatAYPX(r,-1.0,b,UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}
/*@
   PCMGGetCoarseSolve - Gets the solver context to be used on the coarse grid.

   Not Collective

   Input Parameter:
.  pc - the multigrid context

   Output Parameter:
.  ksp - the coarse grid solver context

   Level: advanced

.seealso: `PCMGGetSmootherUp()`, `PCMGGetSmootherDown()`, `PCMGGetSmoother()`
@*/
PetscErrorCode  PCMGGetCoarseSolve(PC pc,KSP *ksp)
{
  PC_MG        *mg        = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *ksp =  mglevels[0]->smoothd;
  PetscFunctionReturn(0);
}

/*@C
   PCMGSetResidual - Sets the function to be used to calculate the residual
   on the lth level.

   Logically Collective on PC

   Input Parameters:
+  pc       - the multigrid context
.  l        - the level (0 is coarsest) to supply
.  residual - function used to form residual, if none is provided the previously provide one is used, if no
              previous one were provided then a default is used
-  mat      - matrix associated with residual

   Level: advanced

.seealso: `PCMGResidualDefault()`
@*/
PetscErrorCode  PCMGSetResidual(PC pc,PetscInt l,PetscErrorCode (*residual)(Mat,Vec,Vec,Vec),Mat mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (residual) mglevels[l]->residual = residual;
  if (!mglevels[l]->residual) mglevels[l]->residual = PCMGResidualDefault;
  mglevels[l]->matresidual = PCMGMatResidualDefault;
  if (mat) PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&mglevels[l]->A));
  mglevels[l]->A = mat;
  PetscFunctionReturn(0);
}

/*@C
   PCMGSetResidualTranspose - Sets the function to be used to calculate the residual of the transposed linear system
   on the lth level.

   Logically Collective on PC

   Input Parameters:
+  pc        - the multigrid context
.  l         - the level (0 is coarsest) to supply
.  residualt - function used to form transpose of residual, if none is provided the previously provide one is used, if no
               previous one were provided then a default is used
-  mat       - matrix associated with residual

   Level: advanced

.seealso: `PCMGResidualTransposeDefault()`
@*/
PetscErrorCode  PCMGSetResidualTranspose(PC pc,PetscInt l,PetscErrorCode (*residualt)(Mat,Vec,Vec,Vec),Mat mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  if (residualt) mglevels[l]->residualtranspose = residualt;
  if (!mglevels[l]->residualtranspose) mglevels[l]->residualtranspose = PCMGResidualTransposeDefault;
  mglevels[l]->matresidualtranspose = PCMGMatResidualTransposeDefault;
  if (mat) PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&mglevels[l]->A));
  mglevels[l]->A = mat;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetInterpolation - Sets the function to be used to calculate the
   interpolation from l-1 to the lth level

   Logically Collective on PC

   Input Parameters:
+  pc  - the multigrid context
.  mat - the interpolation operator
-  l   - the level (0 is coarsest) to supply [do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

.seealso: `PCMGSetRestriction()`
@*/
PetscErrorCode  PCMGSetInterpolation(PC pc,PetscInt l,Mat mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Do not set interpolation routine for coarsest level");
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&mglevels[l]->interpolate));

  mglevels[l]->interpolate = mat;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetOperators - Sets operator and preconditioning matrix for lth level

   Logically Collective on PC

   Input Parameters:
+  pc  - the multigrid context
.  Amat - the operator
.  pmat - the preconditioning operator
-  l   - the level (0 is the coarsest) to supply

   Level: advanced

.keywords:  multigrid, set, interpolate, level

.seealso: `PCMGSetRestriction()`, `PCMGSetInterpolation()`
@*/
PetscErrorCode  PCMGSetOperators(PC pc,PetscInt l,Mat Amat,Mat Pmat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,3);
  PetscValidHeaderSpecific(Pmat,MAT_CLASSID,4);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCall(KSPSetOperators(mglevels[l]->smoothd,Amat,Pmat));
  PetscFunctionReturn(0);
}

/*@
   PCMGGetInterpolation - Gets the function to be used to calculate the
   interpolation from l-1 to the lth level

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Output Parameter:
.  mat - the interpolation matrix, can be NULL

   Level: advanced

.seealso: `PCMGGetRestriction()`, `PCMGSetInterpolation()`, `PCMGGetRScale()`
@*/
PetscErrorCode  PCMGGetInterpolation(PC pc,PetscInt l,Mat *mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat) PetscValidPointer(mat,3);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l > 0 && l < mg->nlevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Level %" PetscInt_FMT " must be in range {1,...,%" PetscInt_FMT "}",l,mg->nlevels-1);
  if (!mglevels[l]->interpolate && mglevels[l]->restrct) {
    PetscCall(PCMGSetInterpolation(pc,l,mglevels[l]->restrct));
  }
  if (mat) *mat = mglevels[l]->interpolate;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetRestriction - Sets the function to be used to restrict dual vectors
   from level l to l-1.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
.  l - the level (0 is coarsest) to supply [Do not supply 0]
-  mat - the restriction matrix

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the interpolation
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with PCMGSetInterpolation()
    is used.

.seealso: `PCMGSetInterpolation()`
@*/
PetscErrorCode  PCMGSetRestriction(PC pc,PetscInt l,Mat mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Do not set restriction routine for coarsest level");
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&mglevels[l]->restrct));

  mglevels[l]->restrct = mat;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetRestriction - Gets the function to be used to restrict dual vectors
   from level l to l-1.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Output Parameter:
.  mat - the restriction matrix

   Level: advanced

.seealso: `PCMGGetInterpolation()`, `PCMGSetRestriction()`, `PCMGGetRScale()`, `PCMGGetInjection()`
@*/
PetscErrorCode  PCMGGetRestriction(PC pc,PetscInt l,Mat *mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat) PetscValidPointer(mat,3);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l > 0 && l < mg->nlevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Level %" PetscInt_FMT " must be in range {1,...,%" PetscInt_FMT "}",l,mg->nlevels-1);
  if (!mglevels[l]->restrct && mglevels[l]->interpolate) {
    PetscCall(PCMGSetRestriction(pc,l,mglevels[l]->interpolate));
  }
  if (mat) *mat = mglevels[l]->restrct;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetRScale - Sets the pointwise scaling for the restriction operator from level l to l-1.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply [Do not supply 0]
.  rscale - the scaling

   Level: advanced

   Notes:
       When evaluating a function on a coarse level one does not want to do F(R * x) one does F(rscale * R * x) where rscale is 1 over the row sums of R.  It is preferable to use PCMGSetInjection() to control moving primal vectors.

.seealso: `PCMGSetInterpolation()`, `PCMGSetRestriction()`, `PCMGGetRScale()`, `PCMGSetInjection()`
@*/
PetscErrorCode  PCMGSetRScale(PC pc,PetscInt l,Vec rscale)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l > 0 && l < mg->nlevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Level %" PetscInt_FMT " must be in range {1,...,%" PetscInt_FMT "}",l,mg->nlevels-1);
  PetscCall(PetscObjectReference((PetscObject)rscale));
  PetscCall(VecDestroy(&mglevels[l]->rscale));

  mglevels[l]->rscale = rscale;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetRScale - Gets the pointwise scaling for the restriction operator from level l to l-1.

   Collective on PC

   Input Parameters:
+  pc - the multigrid context
.  rscale - the scaling
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
       When evaluating a function on a coarse level one does not want to do F(R * x) one does F(rscale * R * x) where rscale is 1 over the row sums of R.  It is preferable to use PCMGGetInjection() to control moving primal vectors.

.seealso: `PCMGSetInterpolation()`, `PCMGGetRestriction()`, `PCMGGetInjection()`
@*/
PetscErrorCode PCMGGetRScale(PC pc,PetscInt l,Vec *rscale)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l > 0 && l < mg->nlevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Level %" PetscInt_FMT " must be in range {1,...,%" PetscInt_FMT "}",l,mg->nlevels-1);
  if (!mglevels[l]->rscale) {
    Mat      R;
    Vec      X,Y,coarse,fine;
    PetscInt M,N;
    PetscCall(PCMGGetRestriction(pc,l,&R));
    PetscCall(MatCreateVecs(R,&X,&Y));
    PetscCall(MatGetSize(R,&M,&N));
    if (M < N) {
      fine = X;
      coarse = Y;
    } else if (N < M) {
      fine = Y; coarse = X;
    } else SETERRQ(PetscObjectComm((PetscObject)R),PETSC_ERR_SUP,"Restriction matrix is square, cannot determine which Vec is coarser");
    PetscCall(VecSet(fine,1.));
    PetscCall(MatRestrict(R,fine,coarse));
    PetscCall(VecDestroy(&fine));
    PetscCall(VecReciprocal(coarse));
    mglevels[l]->rscale = coarse;
  }
  *rscale = mglevels[l]->rscale;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetInjection - Sets the function to be used to inject primal vectors
   from level l to l-1.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
.  l - the level (0 is coarsest) to supply [Do not supply 0]
-  mat - the injection matrix

   Level: advanced

.seealso: `PCMGSetRestriction()`
@*/
PetscErrorCode  PCMGSetInjection(PC pc,PetscInt l,Mat mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Do not set restriction routine for coarsest level");
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&mglevels[l]->inject));

  mglevels[l]->inject = mat;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetInjection - Gets the function to be used to inject primal vectors
   from level l to l-1.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply [Do not supply 0]

   Output Parameter:
.  mat - the restriction matrix (may be NULL if no injection is available).

   Level: advanced

.seealso: `PCMGSetInjection()`, `PCMGetGetRestriction()`
@*/
PetscErrorCode  PCMGGetInjection(PC pc,PetscInt l,Mat *mat)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat) PetscValidPointer(mat,3);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l > 0 && l < mg->nlevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Level %" PetscInt_FMT " must be in range {1,...,%" PetscInt_FMT "}",l,mg->nlevels-1);
  if (mat) *mat = mglevels[l]->inject;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetSmoother - Gets the KSP context to be used as smoother for
   both pre- and post-smoothing.  Call both PCMGGetSmootherUp() and
   PCMGGetSmootherDown() to use different functions for pre- and
   post-smoothing.

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context
-  l - the level (0 is coarsest) to supply

   Output Parameter:
.  ksp - the smoother

   Notes:
   Once you have called this routine, you can call KSPSetOperators(ksp,...) on the resulting ksp to provide the operators for the smoother for this level.
   You can also modify smoother options by calling the various KSPSetXXX() options on this ksp. In addition you can call KSPGetPC(ksp,&pc)
   and modify PC options for the smoother; for example PCSetType(pc,PCSOR); to use SOR smoothing.

   Level: advanced

.seealso: `PCMGGetSmootherUp()`, `PCMGGetSmootherDown()`, `PCMGGetCoarseSolve()`
@*/
PetscErrorCode  PCMGGetSmoother(PC pc,PetscInt l,KSP *ksp)
{
  PC_MG        *mg        = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *ksp = mglevels[l]->smoothd;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetSmootherUp - Gets the KSP context to be used as smoother after
   coarse grid correction (post-smoother).

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context
-  l  - the level (0 is coarsest) to supply

   Output Parameter:
.  ksp - the smoother

   Level: advanced

   Notes:
    calling this will result in a different pre and post smoother so you may need to
         set options on the pre smoother also

.seealso: `PCMGGetSmootherUp()`, `PCMGGetSmootherDown()`
@*/
PetscErrorCode  PCMGGetSmootherUp(PC pc,PetscInt l,KSP *ksp)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  const char     *prefix;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated,
     if not we allocate it.
  */
  PetscCheck(l,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"There is no such thing as a up smoother on the coarse grid");
  if (mglevels[l]->smoothu == mglevels[l]->smoothd) {
    KSPType     ksptype;
    PCType      pctype;
    PC          ipc;
    PetscReal   rtol,abstol,dtol;
    PetscInt    maxits;
    KSPNormType normtype;
    PetscCall(PetscObjectGetComm((PetscObject)mglevels[l]->smoothd,&comm));
    PetscCall(KSPGetOptionsPrefix(mglevels[l]->smoothd,&prefix));
    PetscCall(KSPGetTolerances(mglevels[l]->smoothd,&rtol,&abstol,&dtol,&maxits));
    PetscCall(KSPGetType(mglevels[l]->smoothd,&ksptype));
    PetscCall(KSPGetNormType(mglevels[l]->smoothd,&normtype));
    PetscCall(KSPGetPC(mglevels[l]->smoothd,&ipc));
    PetscCall(PCGetType(ipc,&pctype));

    PetscCall(KSPCreate(comm,&mglevels[l]->smoothu));
    PetscCall(KSPSetErrorIfNotConverged(mglevels[l]->smoothu,pc->erroriffailure));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)mglevels[l]->smoothu,(PetscObject)pc,mglevels[0]->levels-l));
    PetscCall(KSPSetOptionsPrefix(mglevels[l]->smoothu,prefix));
    PetscCall(KSPSetTolerances(mglevels[l]->smoothu,rtol,abstol,dtol,maxits));
    PetscCall(KSPSetType(mglevels[l]->smoothu,ksptype));
    PetscCall(KSPSetNormType(mglevels[l]->smoothu,normtype));
    PetscCall(KSPSetConvergenceTest(mglevels[l]->smoothu,KSPConvergedSkip,NULL,NULL));
    PetscCall(KSPGetPC(mglevels[l]->smoothu,&ipc));
    PetscCall(PCSetType(ipc,pctype));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)mglevels[l]->smoothu));
    PetscCall(PetscObjectComposedDataSetInt((PetscObject) mglevels[l]->smoothu, PetscMGLevelId, mglevels[l]->level));
  }
  if (ksp) *ksp = mglevels[l]->smoothu;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetSmootherDown - Gets the KSP context to be used as smoother before
   coarse grid correction (pre-smoother).

   Not Collective, KSP returned is parallel if PC is

   Input Parameters:
+  pc - the multigrid context
-  l  - the level (0 is coarsest) to supply

   Output Parameter:
.  ksp - the smoother

   Level: advanced

   Notes:
    calling this will result in a different pre and post smoother so you may need to
         set options on the post smoother also

.seealso: `PCMGGetSmootherUp()`, `PCMGGetSmoother()`
@*/
PetscErrorCode  PCMGGetSmootherDown(PC pc,PetscInt l,KSP *ksp)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  /* make sure smoother up and down are different */
  if (l) PetscCall(PCMGGetSmootherUp(pc,l,NULL));
  *ksp = mglevels[l]->smoothd;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetCycleTypeOnLevel - Sets the type of cycle (aka cycle index) to run on the specified level.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
.  l  - the level (0 is coarsest)
-  c  - either PC_MG_CYCLE_V or PC_MG_CYCLE_W

   Level: advanced

.seealso: `PCMGSetCycleType()`
@*/
PetscErrorCode  PCMGSetCycleTypeOnLevel(PC pc,PetscInt l,PCMGCycleType c)
{
  PC_MG        *mg        = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,l,2);
  PetscValidLogicalCollectiveEnum(pc,c,3);
  mglevels[l]->cycles = c;
  PetscFunctionReturn(0);
}

/*@
  PCMGSetRhs - Sets the vector to be used to store the right-hand side on a particular level.

   Logically Collective on PC

  Input Parameters:
+ pc - the multigrid context
. l  - the level (0 is coarsest) this is to be used for
- c  - the Vec

  Level: advanced

  Notes:
  If this is not provided PETSc will automatically generate one. You do not need to keep a reference to this vector if you do not need it. PCDestroy() will properly free it.

.keywords: MG, multigrid, set, right-hand-side, rhs, level
.seealso: `PCMGSetX()`, `PCMGSetR()`
@*/
PetscErrorCode  PCMGSetRhs(PC pc,PetscInt l,Vec c)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l != mglevels[0]->levels-1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Do not set rhs for finest level");
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&mglevels[l]->b));

  mglevels[l]->b = c;
  PetscFunctionReturn(0);
}

/*@
  PCMGSetX - Sets the vector to be used to store the solution on a particular level.

  Logically Collective on PC

  Input Parameters:
+ pc - the multigrid context
. l - the level (0 is coarsest) this is to be used for (do not supply the finest level)
- c - the Vec

  Level: advanced

  Notes:
  If this is not provided PETSc will automatically generate one. You do not need to keep a reference to this vector if you do not need it. PCDestroy() will properly free it.

.keywords: MG, multigrid, set, solution, level
.seealso: `PCMGSetRhs()`, `PCMGSetR()`
@*/
PetscErrorCode  PCMGSetX(PC pc,PetscInt l,Vec c)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l != mglevels[0]->levels-1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Do not set x for finest level");
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&mglevels[l]->x));

  mglevels[l]->x = c;
  PetscFunctionReturn(0);
}

/*@
  PCMGSetR - Sets the vector to be used to store the residual on a particular level.

  Logically Collective on PC

  Input Parameters:
+ pc - the multigrid context
. l - the level (0 is coarsest) this is to be used for
- c - the Vec

  Level: advanced

  Notes:
  If this is not provided PETSc will automatically generate one. You do not need to keep a reference to this vector if you do not need it. PCDestroy() will properly free it.

.keywords: MG, multigrid, set, residual, level
.seealso: `PCMGSetRhs()`, `PCMGSetX()`
@*/
PetscErrorCode  PCMGSetR(PC pc,PetscInt l,Vec c)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscCheck(l,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Need not set residual vector for coarse grid");
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&mglevels[l]->r));

  mglevels[l]->r = c;
  PetscFunctionReturn(0);
}
