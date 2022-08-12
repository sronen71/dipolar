#include <petsc/private/matimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>

PETSC_INTERN PetscErrorCode MatFactorSetUpInPlaceSchur_Private(Mat F)
{
  Mat              St, S = F->schur;
  MatFactorInfo    info;

  PetscFunctionBegin;
  PetscCall(MatSetUnfactored(S));
  PetscCall(MatGetFactor(S,S->solvertype ? S->solvertype : MATSOLVERPETSC,F->factortype,&St));
  if (St->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    PetscCall(MatCholeskyFactorSymbolic(St,S,NULL,&info));
  } else {
    PetscCall(MatLUFactorSymbolic(St,S,NULL,NULL,&info));
  }
  S->ops->solve             = St->ops->solve;
  S->ops->matsolve          = St->ops->matsolve;
  S->ops->solvetranspose    = St->ops->solvetranspose;
  S->ops->matsolvetranspose = St->ops->matsolvetranspose;
  S->ops->solveadd          = St->ops->solveadd;
  S->ops->solvetransposeadd = St->ops->solvetransposeadd;

  PetscCall(MatDestroy(&St));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatFactorUpdateSchurStatus_Private(Mat F)
{
  Mat            S = F->schur;

  PetscFunctionBegin;
  switch(F->schur_status) {
  case MAT_FACTOR_SCHUR_UNFACTORED:
  case MAT_FACTOR_SCHUR_INVERTED:
    if (S) {
      S->ops->solve             = NULL;
      S->ops->matsolve          = NULL;
      S->ops->solvetranspose    = NULL;
      S->ops->matsolvetranspose = NULL;
      S->ops->solveadd          = NULL;
      S->ops->solvetransposeadd = NULL;
      S->factortype             = MAT_FACTOR_NONE;
      PetscCall(PetscFree(S->solvertype));
    }
    break;
  case MAT_FACTOR_SCHUR_FACTORED:
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %d",F->schur_status);
  }
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorFactorizeSchurComplement_Private(Mat F)
{
  MatFactorInfo  info;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_FactorFactS,F,0,0,0));
  if (F->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    PetscCall(MatCholeskyFactor(F->schur,NULL,&info));
  } else {
    PetscCall(MatLUFactor(F->schur,NULL,NULL,&info));
  }
  PetscCall(PetscLogEventEnd(MAT_FactorFactS,F,0,0,0));
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorInvertSchurComplement_Private(Mat F)
{
  Mat S = F->schur;

  PetscFunctionBegin;
  if (S) {
    PetscMPIInt    size;
    PetscBool      isdense,isdensecuda;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)S),&size));
    PetscCheck(size <= 1,PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not yet implemented");
    PetscCall(PetscObjectTypeCompare((PetscObject)S,MATSEQDENSE,&isdense));
    PetscCall(PetscObjectTypeCompare((PetscObject)S,MATSEQDENSECUDA,&isdensecuda));
    PetscCall(PetscLogEventBegin(MAT_FactorInvS,F,0,0,0));
    if (isdense) {
      PetscCall(MatSeqDenseInvertFactors_Private(S));
#if defined(PETSC_HAVE_CUDA)
    } else if (isdensecuda) {
      PetscCall(MatSeqDenseCUDAInvertFactors_Private(S));
#endif
    } else SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not implemented for type %s",((PetscObject)S)->type_name);
    PetscCall(PetscLogEventEnd(MAT_FactorInvS,F,0,0,0));
  }
  PetscFunctionReturn(0);
}
