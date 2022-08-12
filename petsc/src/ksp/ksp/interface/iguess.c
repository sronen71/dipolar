#include <petsc/private/kspimpl.h> /*I "petscksp.h"  I*/

PetscFunctionList KSPGuessList = NULL;
static PetscBool KSPGuessRegisterAllCalled;

/*
  KSPGuessRegister -  Adds a method for initial guess computation in Krylov subspace solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   KSPGuessRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   KSPGuessRegister("my_initial_guess",MyInitialGuessCreate);
.ve

   Then, it can be chosen with the procedural interface via
$     KSPSetGuessType(ksp,"my_initial_guess")
   or at runtime via the option
$     -ksp_guess_type my_initial_guess

   Level: advanced

.seealso: `KSPGuess`, `KSPGuessRegisterAll()`

@*/
PetscErrorCode  KSPGuessRegister(const char sname[],PetscErrorCode (*function)(KSPGuess))
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscFunctionListAdd(&KSPGuessList,sname,function));
  PetscFunctionReturn(0);
}

/*
  KSPGuessRegisterAll - Registers all KSPGuess implementations in the KSP package.

  Not Collective

  Level: advanced

.seealso: `KSPRegisterAll()`, `KSPInitializePackage()`
*/
PetscErrorCode KSPGuessRegisterAll(void)
{
  PetscFunctionBegin;
  if (KSPGuessRegisterAllCalled) PetscFunctionReturn(0);
  KSPGuessRegisterAllCalled = PETSC_TRUE;
  PetscCall(KSPGuessRegister(KSPGUESSFISCHER,KSPGuessCreate_Fischer));
  PetscCall(KSPGuessRegister(KSPGUESSPOD,KSPGuessCreate_POD));
  PetscFunctionReturn(0);
}

/*@
    KSPGuessSetFromOptions - Sets the options for a KSPGuess from the options database

    Collective on guess

    Input Parameter:
.    guess - KSPGuess object

   Level: intermediate

.seealso: `KSPGuess`, `KSPGetGuess()`, `KSPSetGuessType()`, `KSPGuessType`
@*/
PetscErrorCode KSPGuessSetFromOptions(KSPGuess guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (guess->ops->setfromoptions) PetscCall((*guess->ops->setfromoptions)(guess));
  PetscFunctionReturn(0);
}

/*@
    KSPGuessSetTolerance - Sets the relative tolerance used in either eigenvalue (POD) or singular value (Fischer type 3) calculations. Ignored by the first and second Fischer types.

    Collective on guess

    Input Parameter:
.    guess - KSPGuess object

   Level: intermediate

.seealso: `KSPGuess`, `KSPGuessType`, `KSPGuessSetFromOptions()`
@*/
PetscErrorCode KSPGuessSetTolerance(KSPGuess guess, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (guess->ops->settolerance) PetscCall((*guess->ops->settolerance)(guess,tol));
  PetscFunctionReturn(0);
}

/*@
   KSPGuessDestroy - Destroys KSPGuess context.

   Collective on kspGuess

   Input Parameter:
.  guess - initial guess object

   Level: beginner

.seealso: `KSPGuessCreate()`, `KSPGuess`, `KSPGuessType`
@*/
PetscErrorCode  KSPGuessDestroy(KSPGuess *guess)
{
  PetscFunctionBegin;
  if (!*guess) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*guess),KSPGUESS_CLASSID,1);
  if (--((PetscObject)(*guess))->refct > 0) {*guess = NULL; PetscFunctionReturn(0);}
  if ((*guess)->ops->destroy) PetscCall((*(*guess)->ops->destroy)(*guess));
  PetscCall(MatDestroy(&(*guess)->A));
  PetscCall(PetscHeaderDestroy(guess));
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessView - View the KSPGuess object

   Logically Collective on guess

   Input Parameters:
+  guess  - the initial guess object for the Krylov method
-  viewer - the viewer object

   Notes:

  Level: intermediate

.seealso: `KSP`, `KSPGuess`, `KSPGuessType`, `KSPGuessRegister()`, `KSPGuessCreate()`, `PetscViewer`
@*/
PetscErrorCode  KSPGuessView(KSPGuess guess, PetscViewer view)
{
  PetscBool      ascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (!view) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)guess),&view));
  }
  PetscValidHeaderSpecific(view,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(guess,1,view,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&ascii));
  if (ascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)guess,view));
    if (guess->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(view));
      PetscCall((*guess->ops->view)(guess,view));
      PetscCall(PetscViewerASCIIPopTab(view));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   KSPGuessCreate - Creates the default KSPGuess context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  guess - location to put the KSPGuess context

   Notes:
   The default KSPGuess type is XXX

   Level: beginner

.seealso: `KSPSolve()`, `KSPGuessDestroy()`, `KSPGuess`, `KSPGuessType`, `KSP`
@*/
PetscErrorCode  KSPGuessCreate(MPI_Comm comm,KSPGuess *guess)
{
  KSPGuess       tguess;

  PetscFunctionBegin;
  PetscValidPointer(guess,2);
  *guess = NULL;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscHeaderCreate(tguess,KSPGUESS_CLASSID,"KSPGuess","Initial guess for Krylov Method","KSPGuess",comm,KSPGuessDestroy,KSPGuessView));
  tguess->omatstate = -1;
  *guess = tguess;
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessSetType - Sets the type of a KSPGuess

   Logically Collective on guess

   Input Parameters:
+  guess - the initial guess object for the Krylov method
-  type  - a known KSPGuess method

   Options Database Key:
.  -ksp_guess_type  <method> - Sets the method; use -help for a list
    of available methods

   Notes:

  Level: intermediate

.seealso: `KSP`, `KSPGuess`, `KSPGuessType`, `KSPGuessRegister()`, `KSPGuessCreate()`

@*/
PetscErrorCode  KSPGuessSetType(KSPGuess guess, KSPGuessType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(KSPGuess);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)guess,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(KSPGuessList,type,&r));
  PetscCheck(r,PetscObjectComm((PetscObject)guess),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested KSPGuess type %s",type);
  if (guess->ops->destroy) {
    PetscCall((*guess->ops->destroy)(guess));
    guess->ops->destroy = NULL;
  }
  PetscCall(PetscMemzero(guess->ops,sizeof(struct _KSPGuessOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)guess,type));
  PetscCall((*r)(guess));
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessGetType - Gets the KSPGuess type as a string from the KSPGuess object.

   Not Collective

   Input Parameter:
.  guess - the initial guess context

   Output Parameter:
.  name - name of KSPGuess method

   Level: intermediate

.seealso: `KSPGuessSetType()`
@*/
PetscErrorCode  KSPGuessGetType(KSPGuess guess,KSPGuessType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)guess)->type_name;
  PetscFunctionReturn(0);
}

/*@
    KSPGuessUpdate - Updates the guess object with the current solution and rhs vector

   Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  rhs   - the corresponding rhs
-  sol   - the computed solution

   Level: intermediate

.seealso: `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode  KSPGuessUpdate(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  if (guess->ops->update) PetscCall((*guess->ops->update)(guess,rhs,sol));
  PetscFunctionReturn(0);
}

/*@
    KSPGuessFormGuess - Form the initial guess

   Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  rhs   - the current rhs vector
-  sol   - the initial guess vector

   Level: intermediate

.seealso: `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode  KSPGuessFormGuess(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  if (guess->ops->formguess) PetscCall((*guess->ops->formguess)(guess,rhs,sol));
  PetscFunctionReturn(0);
}

/*@
    KSPGuessSetUp - Setup the initial guess object

   Collective on guess

   Input Parameter:
-  guess - the initial guess context

   Level: intermediate

.seealso: `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode  KSPGuessSetUp(KSPGuess guess)
{
  PetscObjectState matstate;
  PetscInt         oM = 0, oN = 0, M, N;
  Mat              omat = NULL;
  PC               pc;
  PetscBool        reuse;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (guess->A) {
    omat = guess->A;
    PetscCall(MatGetSize(guess->A,&oM,&oN));
  }
  PetscCall(KSPGetOperators(guess->ksp,&guess->A,NULL));
  PetscCall(KSPGetPC(guess->ksp,&pc));
  PetscCall(PCGetReusePreconditioner(pc,&reuse));
  PetscCall(PetscObjectReference((PetscObject)guess->A));
  PetscCall(MatGetSize(guess->A,&M,&N));
  PetscCall(PetscObjectStateGet((PetscObject)guess->A,&matstate));
  if (M != oM || N != oN) {
    PetscCall(PetscInfo(guess,"Resetting KSPGuess since matrix sizes have changed (%" PetscInt_FMT " != %" PetscInt_FMT ", %" PetscInt_FMT " != %" PetscInt_FMT ")\n",oM,M,oN,N));
  } else if (!reuse && (omat != guess->A || guess->omatstate != matstate)) {
    PetscCall(PetscInfo(guess,"Resetting KSPGuess since %s has changed\n",omat != guess->A ? "matrix" : "matrix state"));
    if (guess->ops->reset) PetscCall((*guess->ops->reset)(guess));
  } else if (reuse) {
    PetscCall(PetscInfo(guess,"Not resettting KSPGuess since reuse preconditioner has been specified\n"));
  } else {
    PetscCall(PetscInfo(guess,"KSPGuess status unchanged\n"));
  }
  if (guess->ops->setup) PetscCall((*guess->ops->setup)(guess));
  guess->omatstate = matstate;
  PetscCall(MatDestroy(&omat));
  PetscFunctionReturn(0);
}
