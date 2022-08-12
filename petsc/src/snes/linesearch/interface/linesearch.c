#include <petsc/private/linesearchimpl.h> /*I "petscsnes.h" I*/

PetscBool         SNESLineSearchRegisterAllCalled = PETSC_FALSE;
PetscFunctionList SNESLineSearchList              = NULL;

PetscClassId  SNESLINESEARCH_CLASSID;
PetscLogEvent SNESLINESEARCH_Apply;

/*@
   SNESLineSearchMonitorCancel - Clears all the monitor functions for a SNESLineSearch object.

   Logically Collective on SNESLineSearch

   Input Parameters:
.  ls - the SNESLineSearch context

   Options Database Key:
.  -snes_linesearch_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to SNESLineSearchMonitorSet(), but does not cancel those
    set via the options database

   Notes:
   There is no way to clear one specific monitor from a SNESLineSearch object.

   This does not clear the monitor set with SNESLineSearchSetDefaultMonitor() use SNESLineSearchSetDefaultMonitor(ls,NULL) to cancel
   that one.

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESLineSearchMonitorDefault()`, `SNESLineSearchMonitorSet()`
@*/
PetscErrorCode  SNESLineSearchMonitorCancel(SNESLineSearch ls)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,SNESLINESEARCH_CLASSID,1);
  for (i=0; i<ls->numbermonitors; i++) {
    if (ls->monitordestroy[i]) {
      PetscCall((*ls->monitordestroy[i])(&ls->monitorcontext[i]));
    }
  }
  ls->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchMonitor - runs the user provided monitor routines, if they exist

   Collective on SNES

   Input Parameters:
.  ls - the linesearch object

   Notes:
   This routine is called by the SNES implementations.
   It does not typically need to be called by the user.

   Level: developer

.seealso: `SNESGetLineSearch()`, `SNESLineSearchMonitorSet()`
@*/
PetscErrorCode  SNESLineSearchMonitor(SNESLineSearch ls)
{
  PetscInt       i,n = ls->numbermonitors;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    PetscCall((*ls->monitorftns[i])(ls,ls->monitorcontext[i]));
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchMonitorSet - Sets an ADDITIONAL function that is to be used at every
   iteration of the nonlinear solver to display the iteration's
   progress.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  ls - the SNESLineSearch context
.  f - the monitor function
.  mctx - [optional] user-defined context for private data for the
          monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Notes:
   Several different monitoring routines may be set by calling
   SNESLineSearchMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Fortran Notes:
    Only a single monitor function can be set for each SNESLineSearch object

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESLineSearchMonitorDefault()`, `SNESLineSearchMonitorCancel()`
@*/
PetscErrorCode  SNESLineSearchMonitorSet(SNESLineSearch ls,PetscErrorCode (*f)(SNESLineSearch,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,SNESLINESEARCH_CLASSID,1);
  for (i=0; i<ls->numbermonitors;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))f,mctx,monitordestroy,(PetscErrorCode (*)(void))ls->monitorftns[i],ls->monitorcontext[i],ls->monitordestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(ls->numbermonitors < MAXSNESLSMONITORS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  ls->monitorftns[ls->numbermonitors]          = f;
  ls->monitordestroy[ls->numbermonitors]   = monitordestroy;
  ls->monitorcontext[ls->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchMonitorSolutionUpdate - Monitors each update a new function value the linesearch tries

   Collective on SNESLineSearch

   Input Parameters:
+  ls - the SNES linesearch object
-  vf - the context for the monitor, in this case it is an ASCII PetscViewer and format

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESMonitorSet()`, `SNESMonitorSolution()`
@*/
PetscErrorCode  SNESLineSearchMonitorSolutionUpdate(SNESLineSearch ls,PetscViewerAndFormat *vf)
{
  PetscViewer    viewer = vf->viewer;
  Vec            Y,W,G;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(ls,NULL,NULL,&Y,&W,&G));
  PetscCall(PetscViewerPushFormat(viewer,vf->format));
  PetscCall(PetscViewerASCIIPrintf(viewer,"LineSearch attempted update to solution \n"));
  PetscCall(VecView(Y,viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"LineSearch attempted new solution \n"));
  PetscCall(VecView(W,viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"LineSearch attempted updated function value\n"));
  PetscCall(VecView(G,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchCreate - Creates the line search context.

   Logically Collective on Comm

   Input Parameters:
.  comm - MPI communicator for the line search (typically from the associated SNES context).

   Output Parameters:
.  outlinesearch - the new linesearch context

   Level: developer

   Notes:
   The preferred calling sequence for users is to use SNESGetLineSearch() to acquire the SNESLineSearch instance
   already associated with the SNES.  This function is for developer use.

.seealso: `LineSearchDestroy()`, `SNESGetLineSearch()`
@*/

PetscErrorCode SNESLineSearchCreate(MPI_Comm comm, SNESLineSearch *outlinesearch)
{
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  PetscValidPointer(outlinesearch,2);
  PetscCall(SNESInitializePackage());
  *outlinesearch = NULL;

  PetscCall(PetscHeaderCreate(linesearch,SNESLINESEARCH_CLASSID, "SNESLineSearch","Linesearch","SNESLineSearch",comm,SNESLineSearchDestroy,SNESLineSearchView));

  linesearch->vec_sol_new  = NULL;
  linesearch->vec_func_new = NULL;
  linesearch->vec_sol      = NULL;
  linesearch->vec_func     = NULL;
  linesearch->vec_update   = NULL;

  linesearch->lambda       = 1.0;
  linesearch->fnorm        = 1.0;
  linesearch->ynorm        = 1.0;
  linesearch->xnorm        = 1.0;
  linesearch->result       = SNES_LINESEARCH_SUCCEEDED;
  linesearch->norms        = PETSC_TRUE;
  linesearch->keeplambda   = PETSC_FALSE;
  linesearch->damping      = 1.0;
  linesearch->maxstep      = 1e8;
  linesearch->steptol      = 1e-12;
  linesearch->rtol         = 1e-8;
  linesearch->atol         = 1e-15;
  linesearch->ltol         = 1e-8;
  linesearch->precheckctx  = NULL;
  linesearch->postcheckctx = NULL;
  linesearch->max_its      = 1;
  linesearch->setupcalled  = PETSC_FALSE;
  linesearch->monitor      = NULL;
  *outlinesearch           = linesearch;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetUp - Prepares the line search for being applied by allocating
   any required vectors.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The LineSearch instance.

   Notes:
   For most cases, this needn't be called by users or outside of SNESLineSearchApply().
   The only current case where this is called outside of this is for the VI
   solvers, which modify the solution and work vectors before the first call
   of SNESLineSearchApply, requiring the SNESLineSearch work vectors to be
   allocated upfront.

   Level: advanced

.seealso: `SNESGetLineSearch()`, `SNESLineSearchReset()`
@*/

PetscErrorCode SNESLineSearchSetUp(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC));
  }
  if (!linesearch->setupcalled) {
    if (!linesearch->vec_sol_new) {
      PetscCall(VecDuplicate(linesearch->vec_sol, &linesearch->vec_sol_new));
    }
    if (!linesearch->vec_func_new) {
      PetscCall(VecDuplicate(linesearch->vec_sol, &linesearch->vec_func_new));
    }
    if (linesearch->ops->setup) PetscCall((*linesearch->ops->setup)(linesearch));
    if (!linesearch->ops->snesfunc) PetscCall(SNESLineSearchSetFunction(linesearch,SNESComputeFunction));
    linesearch->lambda      = linesearch->damping;
    linesearch->setupcalled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchReset - Undoes the SNESLineSearchSetUp() and deletes any Vecs or Mats allocated by the line search.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The LineSearch instance.

   Notes:
    Usually only called by SNESReset()

   Level: developer

.seealso: `SNESGetLineSearch()`, `SNESLineSearchSetUp()`
@*/

PetscErrorCode SNESLineSearchReset(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  if (linesearch->ops->reset) PetscCall((*linesearch->ops->reset)(linesearch));

  PetscCall(VecDestroy(&linesearch->vec_sol_new));
  PetscCall(VecDestroy(&linesearch->vec_func_new));

  PetscCall(VecDestroyVecs(linesearch->nwork, &linesearch->work));

  linesearch->nwork       = 0;
  linesearch->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetFunction - Sets the function evaluation used by the SNES line search

   Input Parameters:
.  linesearch - the SNESLineSearch context
+  func       - function evaluation routine

   Level: developer

   Notes:
    This is used internally by PETSc and not called by users

.seealso: `SNESGetLineSearch()`, `SNESSetFunction()`
@*/
PetscErrorCode  SNESLineSearchSetFunction(SNESLineSearch linesearch, PetscErrorCode (*func)(SNES,Vec,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->ops->snesfunc = func;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetPreCheck - Sets a user function that is called after the initial search direction has been computed but
         before the line search routine has been applied. Allows the user to adjust the result of (usually a linear solve) that
         determined the search direction.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - the SNESLineSearch context
.  func - [optional] function evaluation routine, see SNESLineSearchPreCheck() for the calling sequence
-  ctx        - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Notes:
   Use `SNESLineSearchSetPostCheck()` to change the step after the line search.
   search is complete.

   Use `SNESVISetVariableBounds()` and `SNESVISetComputeVariableBounds()` to cause `SNES` to automatically control the ranges of variables allowed.

.seealso: `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchGetPreCheck()`,
          `SNESVISetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESSetFunctionDomainError()`, `SNESSetJacobianDomainError()

@*/
PetscErrorCode  SNESLineSearchSetPreCheck(SNESLineSearch linesearch, PetscErrorCode (*func)(SNESLineSearch,Vec,Vec,PetscBool*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) linesearch->ops->precheck = func;
  if (ctx) linesearch->precheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchGetPreCheck - Gets the pre-check function for the line search routine.

   Input Parameter:
.  linesearch - the SNESLineSearch context

   Output Parameters:
+  func       - [optional] function evaluation routine, see SNESLineSearchPreCheck() for calling sequence
-  ctx        - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchSetPostCheck()`
@*/
PetscErrorCode  SNESLineSearchGetPreCheck(SNESLineSearch linesearch, PetscErrorCode (**func)(SNESLineSearch,Vec,Vec,PetscBool*,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) *func = linesearch->ops->precheck;
  if (ctx) *ctx = linesearch->precheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetPostCheck - Sets a user function that is called after the line search has been applied to determine the step
       direction and length. Allows the user a chance to change or override the decision of the line search routine

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - the SNESLineSearch context
.  func - [optional] function evaluation routine, see SNESLineSearchPostCheck()  for the calling sequence
-  ctx        - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Notes:
   Use `SNESLineSearchSetPreCheck()` to change the step before the line search.
   search is complete.

   Use `SNESVISetVariableBounds()` and `SNESVISetComputeVariableBounds()` to cause `SNES` to automatically control the ranges of variables allowed.

.seealso: `SNESGetLineSearch()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchGetPostCheck()`,
          `SNESVISetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESSetFunctionDomainError()`, `SNESSetJacobianDomainError()
@*/
PetscErrorCode  SNESLineSearchSetPostCheck(SNESLineSearch linesearch, PetscErrorCode (*func)(SNESLineSearch,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) linesearch->ops->postcheck = func;
  if (ctx) linesearch->postcheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchGetPostCheck - Gets the post-check function for the line search routine.

   Input Parameter:
.  linesearch - the SNESLineSearch context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESLineSearchPostCheck()
-  ctx        - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`
@*/
PetscErrorCode  SNESLineSearchGetPostCheck(SNESLineSearch linesearch, PetscErrorCode (**func)(SNESLineSearch,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (func) *func = linesearch->ops->postcheck;
  if (ctx) *ctx = linesearch->postcheckctx;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchPreCheck - Prepares the line search for being applied.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch instance.
.  X - The current solution
-  Y - The step direction

   Output Parameters:
.  changed - Indicator that the precheck routine has changed anything

   Level: advanced

.seealso: `SNESGetLineSearch()`, `SNESLineSearchPostCheck()`, `SNESLineSearchSetPreCheck()`, `SNESLineSearchGetPreCheck()`, `SNESLineSearchSetPostCheck()`,
          `SNESLineSearchGetPostCheck()``
@*/
PetscErrorCode SNESLineSearchPreCheck(SNESLineSearch linesearch,Vec X,Vec Y,PetscBool *changed)
{
  PetscFunctionBegin;
  *changed = PETSC_FALSE;
  if (linesearch->ops->precheck) {
    PetscCall((*linesearch->ops->precheck)(linesearch, X, Y, changed, linesearch->precheckctx));
    PetscValidLogicalCollectiveBool(linesearch,*changed,4);
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchPostCheck - Hook to modify step direction or updated solution after a successful linesearch

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch context
.  X - The last solution
.  Y - The step direction
-  W - The updated solution, W = X + lambda*Y for some lambda

   Output Parameters:
+  changed_Y - Indicator if the direction Y has been changed.
-  changed_W - Indicator if the new candidate solution W has been changed.

   Level: developer

.seealso: `SNESGetLineSearch()`, `SNESLineSearchPreCheck()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchSetPrecheck()`, `SNESLineSearchGetPrecheck()`
@*/
PetscErrorCode SNESLineSearchPostCheck(SNESLineSearch linesearch,Vec X,Vec Y,Vec W,PetscBool *changed_Y,PetscBool *changed_W)
{
  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (linesearch->ops->postcheck) {
    PetscCall((*linesearch->ops->postcheck)(linesearch,X,Y,W,changed_Y,changed_W,linesearch->postcheckctx));
    PetscValidLogicalCollectiveBool(linesearch,*changed_Y,5);
    PetscValidLogicalCollectiveBool(linesearch,*changed_W,6);
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchPreCheckPicard - Implements a correction that is sometimes useful to improve the convergence rate of Picard iteration

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - linesearch context
.  X - base state for this step
-  ctx - context for this function

   Input/Output Parameter:
.  Y - correction, possibly modified

   Output Parameter:
.  changed - flag indicating that Y was modified

   Options Database Key:
+  -snes_linesearch_precheck_picard - activate this routine
-  -snes_linesearch_precheck_picard_angle - angle

   Level: advanced

   Notes:
   This function should be passed to SNESLineSearchSetPreCheck()

   The justification for this method involves the linear convergence of a Picard iteration
   so the Picard linearization should be provided in place of the "Jacobian". This correction
   is generally not useful when using a Newton linearization.

   Reference:
   Hindmarsh and Payne (1996) Time step limits for stable solutions of the ice sheet equation, Annals of Glaciology.

.seealso: `SNESGetLineSearch()`, `SNESLineSearchSetPreCheck()`
@*/
PetscErrorCode SNESLineSearchPreCheckPicard(SNESLineSearch linesearch,Vec X,Vec Y,PetscBool *changed,void *ctx)
{
  PetscReal      angle = *(PetscReal*)linesearch->precheckctx;
  Vec            Ylast;
  PetscScalar    dot;
  PetscInt       iter;
  PetscReal      ynorm,ylastnorm,theta,angle_radians;
  SNES           snes;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(PetscObjectQuery((PetscObject)snes,"SNESLineSearchPreCheckPicard_Ylast",(PetscObject*)&Ylast));
  if (!Ylast) {
    PetscCall(VecDuplicate(Y,&Ylast));
    PetscCall(PetscObjectCompose((PetscObject)snes,"SNESLineSearchPreCheckPicard_Ylast",(PetscObject)Ylast));
    PetscCall(PetscObjectDereference((PetscObject)Ylast));
  }
  PetscCall(SNESGetIterationNumber(snes,&iter));
  if (iter < 2) {
    PetscCall(VecCopy(Y,Ylast));
    *changed = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  PetscCall(VecDot(Y,Ylast,&dot));
  PetscCall(VecNorm(Y,NORM_2,&ynorm));
  PetscCall(VecNorm(Ylast,NORM_2,&ylastnorm));
  /* Compute the angle between the vectors Y and Ylast, clip to keep inside the domain of acos() */
  theta         = PetscAcosReal((PetscReal)PetscClipInterval(PetscAbsScalar(dot) / (ynorm * ylastnorm),-1.0,1.0));
  angle_radians = angle * PETSC_PI / 180.;
  if (PetscAbsReal(theta) < angle_radians || PetscAbsReal(theta - PETSC_PI) < angle_radians) {
    /* Modify the step Y */
    PetscReal alpha,ydiffnorm;
    PetscCall(VecAXPY(Ylast,-1.0,Y));
    PetscCall(VecNorm(Ylast,NORM_2,&ydiffnorm));
    alpha = (ydiffnorm > .001*ylastnorm) ? ylastnorm / ydiffnorm : 1000.0;
    PetscCall(VecCopy(Y,Ylast));
    PetscCall(VecScale(Y,alpha));
    PetscCall(PetscInfo(snes,"Angle %14.12e degrees less than threshold %14.12e, corrected step by alpha=%14.12e\n",(double)(theta*180./PETSC_PI),(double)angle,(double)alpha));
    *changed = PETSC_TRUE;
  } else {
    PetscCall(PetscInfo(snes,"Angle %14.12e degrees exceeds threshold %14.12e, no correction applied\n",(double)(theta*180./PETSC_PI),(double)angle));
    PetscCall(VecCopy(Y,Ylast));
    *changed = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchApply - Computes the line-search update.

   Collective on SNESLineSearch

   Input Parameters:
+  linesearch - The linesearch context
-  Y - The search direction

   Input/Output Parameters:
+  X - The current solution, on output the new solution
.  F - The current function, on output the new function
-  fnorm - The current norm, on output the new function norm

   Options Database Keys:
+ -snes_linesearch_type - basic (or equivalently none), bt, l2, cp, nleqerr, shell
. -snes_linesearch_monitor [:filename] - Print progress of line searches
. -snes_linesearch_damping - The linesearch damping parameter, default is 1.0 (no damping)
. -snes_linesearch_norms   - Turn on/off the linesearch norms computation (SNESLineSearchSetComputeNorms())
. -snes_linesearch_keeplambda - Keep the previous search length as the initial guess
- -snes_linesearch_max_it - The number of iterations for iterative line searches

   Notes:
   This is typically called from within a SNESSolve() implementation in order to
   help with convergence of the nonlinear method.  Various SNES types use line searches
   in different ways, but the overarching theme is that a line search is used to determine
   an optimal damping parameter of a step at each iteration of the method.  Each
   application of the line search may invoke SNESComputeFunction() several times, and
   therefore may be fairly expensive.

   Level: Intermediate

.seealso: `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchPreCheck()`, `SNESLineSearchPostCheck()`, `SNESSolve()`, `SNESComputeFunction()`, `SNESLineSearchSetComputeNorms()`,
          `SNESLineSearchType`, `SNESLineSearchSetType()`
@*/
PetscErrorCode SNESLineSearchApply(SNESLineSearch linesearch, Vec X, Vec F, PetscReal * fnorm, Vec Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,5);

  linesearch->result = SNES_LINESEARCH_SUCCEEDED;

  linesearch->vec_sol    = X;
  linesearch->vec_update = Y;
  linesearch->vec_func   = F;

  PetscCall(SNESLineSearchSetUp(linesearch));

  if (!linesearch->keeplambda) linesearch->lambda = linesearch->damping; /* set the initial guess to lambda */

  if (fnorm) linesearch->fnorm = *fnorm;
  else {
    PetscCall(VecNorm(F, NORM_2, &linesearch->fnorm));
  }

  PetscCall(PetscLogEventBegin(SNESLINESEARCH_Apply,linesearch,X,F,Y));

  PetscCall((*linesearch->ops->apply)(linesearch));

  PetscCall(PetscLogEventEnd(SNESLINESEARCH_Apply,linesearch,X,F,Y));

  if (fnorm) *fnorm = linesearch->fnorm;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchDestroy - Destroys the line search instance.

   Collective on SNESLineSearch

   Input Parameters:
.  linesearch - The linesearch context

   Level: developer

.seealso: `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchReset()`, `SNESDestroy()`
@*/
PetscErrorCode SNESLineSearchDestroy(SNESLineSearch * linesearch)
{
  PetscFunctionBegin;
  if (!*linesearch) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*linesearch),SNESLINESEARCH_CLASSID,1);
  if (--((PetscObject)(*linesearch))->refct > 0) {*linesearch = NULL; PetscFunctionReturn(0);}
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*linesearch));
  PetscCall(SNESLineSearchReset(*linesearch));
  if ((*linesearch)->ops->destroy) (*linesearch)->ops->destroy(*linesearch);
  PetscCall(PetscViewerDestroy(&(*linesearch)->monitor));
  PetscCall(SNESLineSearchMonitorCancel((*linesearch)));
  PetscCall(PetscHeaderDestroy(linesearch));
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetDefaultMonitor - Turns on/off printing useful information and debugging output about the line search.

   Input Parameters:
+  linesearch - the linesearch object
-  viewer - an ASCII PetscViewer or NULL to turn off monitor

   Logically Collective on SNESLineSearch

   Options Database:
.   -snes_linesearch_monitor [:filename] - enables the monitor

   Level: intermediate

   Developer Note: This monitor is implemented differently than the other SNESLineSearchMonitors that are set with
     SNESLineSearchMonitorSet() since it is called in many locations of the line search routines to display aspects of the
     line search that are not visible to the other monitors.

.seealso: `SNESGetLineSearch()`, `SNESLineSearchGetDefaultMonitor()`, `PetscViewer`, `SNESLineSearchSetMonitor()`
@*/
PetscErrorCode  SNESLineSearchSetDefaultMonitor(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (viewer) PetscCall(PetscObjectReference((PetscObject)viewer));
  PetscCall(PetscViewerDestroy(&linesearch->monitor));
  linesearch->monitor = viewer;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetDefaultMonitor - Gets the PetscViewer instance for the line search monitor.

   Input Parameter:
.  linesearch - linesearch context

   Output Parameter:
.  monitor - monitor context

   Logically Collective on SNES

   Options Database Keys:
.   -snes_linesearch_monitor - enables the monitor

   Level: intermediate

.seealso: `SNESGetLineSearch()`, `SNESLineSearchSetDefaultMonitor()`, `PetscViewer`
@*/
PetscErrorCode  SNESLineSearchGetDefaultMonitor(SNESLineSearch linesearch, PetscViewer *monitor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  *monitor = linesearch->monitor;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on SNESLineSearch

   Input Parameters:
+  ls - LineSearch object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the SNESLineSearch or PetscViewer objects

   Level: developer

.seealso: `PetscOptionsGetViewer()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode  SNESLineSearchMonitorSetFromOptions(SNESLineSearch ls,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(SNESLineSearch,PetscViewerAndFormat*),PetscErrorCode (*monitorsetup)(SNESLineSearch,PetscViewerAndFormat*))
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ls),((PetscObject) ls)->options,((PetscObject)ls)->prefix,name,&viewer,&format,&flg));
  if (flg) {
    PetscViewerAndFormat *vf;
    PetscCall(PetscViewerAndFormatCreate(viewer,format,&vf));
    PetscCall(PetscObjectDereference((PetscObject)viewer));
    if (monitorsetup) PetscCall((*monitorsetup)(ls,vf));
    PetscCall(SNESLineSearchMonitorSet(ls,(PetscErrorCode (*)(SNESLineSearch,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetFromOptions - Sets options for the line search

   Input Parameter:
.  linesearch - linesearch context

   Options Database Keys:
+ -snes_linesearch_type <type> - basic (or equivalently none), bt, l2, cp, nleqerr, shell
. -snes_linesearch_order <order> - 1, 2, 3.  Most types only support certain orders (bt supports 2 or 3)
. -snes_linesearch_norms   - Turn on/off the linesearch norms for the basic linesearch typem (SNESLineSearchSetComputeNorms())
. -snes_linesearch_minlambda - The minimum step length
. -snes_linesearch_maxstep - The maximum step size
. -snes_linesearch_rtol - Relative tolerance for iterative line searches
. -snes_linesearch_atol - Absolute tolerance for iterative line searches
. -snes_linesearch_ltol - Change in lambda tolerance for iterative line searches
. -snes_linesearch_max_it - The number of iterations for iterative line searches
. -snes_linesearch_monitor [:filename] - Print progress of line searches
. -snes_linesearch_monitor_solution_update [viewer:filename:format] - view each update tried by line search routine
. -snes_linesearch_damping - The linesearch damping parameter
. -snes_linesearch_keeplambda - Keep the previous search length as the initial guess.
. -snes_linesearch_precheck_picard - Use precheck that speeds up convergence of picard method
- -snes_linesearch_precheck_picard_angle - Angle used in Picard precheck method

   Logically Collective on SNESLineSearch

   Level: intermediate

.seealso: `SNESLineSearchCreate()`, `SNESLineSearchSetOrder()`, `SNESLineSearchSetType()`, `SNESLineSearchSetTolerances()`, `SNESLineSearchSetDamping()`, `SNESLineSearchPreCheckPicard()`,
          `SNESLineSearchType`, `SNESLineSearchSetComputeNorms()`
@*/
PetscErrorCode SNESLineSearchSetFromOptions(SNESLineSearch linesearch)
{
  const char        *deft = SNESLINESEARCHBASIC;
  char              type[256];
  PetscBool         flg, set;
  PetscViewer       viewer;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchRegisterAll());

  PetscObjectOptionsBegin((PetscObject)linesearch);
  if (((PetscObject)linesearch)->type_name) deft = ((PetscObject)linesearch)->type_name;
  PetscCall(PetscOptionsFList("-snes_linesearch_type","Linesearch type","SNESLineSearchSetType",SNESLineSearchList,deft,type,256,&flg));
  if (flg) {
    PetscCall(SNESLineSearchSetType(linesearch,type));
  } else if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch,deft));
  }

  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)linesearch),((PetscObject) linesearch)->options,((PetscObject)linesearch)->prefix,"-snes_linesearch_monitor",&viewer,NULL,&set));
  if (set) {
    PetscCall(SNESLineSearchSetDefaultMonitor(linesearch,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(SNESLineSearchMonitorSetFromOptions(linesearch,"-snes_linesearch_monitor_solution_update","View correction at each iteration","SNESLineSearchMonitorSolutionUpdate",SNESLineSearchMonitorSolutionUpdate,NULL));

  /* tolerances */
  PetscCall(PetscOptionsReal("-snes_linesearch_minlambda","Minimum step length","SNESLineSearchSetTolerances",linesearch->steptol,&linesearch->steptol,NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_maxstep","Maximum step size","SNESLineSearchSetTolerances",linesearch->maxstep,&linesearch->maxstep,NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_rtol","Relative tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->rtol,&linesearch->rtol,NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_atol","Absolute tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->atol,&linesearch->atol,NULL));
  PetscCall(PetscOptionsReal("-snes_linesearch_ltol","Change in lambda tolerance for iterative line search","SNESLineSearchSetTolerances",linesearch->ltol,&linesearch->ltol,NULL));
  PetscCall(PetscOptionsInt("-snes_linesearch_max_it","Maximum iterations for iterative line searches","SNESLineSearchSetTolerances",linesearch->max_its,&linesearch->max_its,NULL));

  /* damping parameters */
  PetscCall(PetscOptionsReal("-snes_linesearch_damping","Line search damping and initial step guess","SNESLineSearchSetDamping",linesearch->damping,&linesearch->damping,NULL));

  PetscCall(PetscOptionsBool("-snes_linesearch_keeplambda","Use previous lambda as damping","SNESLineSearchSetKeepLambda",linesearch->keeplambda,&linesearch->keeplambda,NULL));

  /* precheck */
  PetscCall(PetscOptionsBool("-snes_linesearch_precheck_picard","Use a correction that sometimes improves convergence of Picard iteration","SNESLineSearchPreCheckPicard",flg,&flg,&set));
  if (set) {
    if (flg) {
      linesearch->precheck_picard_angle = 10.; /* correction only active if angle is less than 10 degrees */

      PetscCall(PetscOptionsReal("-snes_linesearch_precheck_picard_angle","Maximum angle at which to activate the correction","none",linesearch->precheck_picard_angle,&linesearch->precheck_picard_angle,NULL));
      PetscCall(SNESLineSearchSetPreCheck(linesearch,SNESLineSearchPreCheckPicard,&linesearch->precheck_picard_angle));
    } else {
      PetscCall(SNESLineSearchSetPreCheck(linesearch,NULL,NULL));
    }
  }
  PetscCall(PetscOptionsInt("-snes_linesearch_order","Order of approximation used in the line search","SNESLineSearchSetOrder",linesearch->order,&linesearch->order,NULL));
  PetscCall(PetscOptionsBool("-snes_linesearch_norms","Compute final norms in line search","SNESLineSearchSetComputeNorms",linesearch->norms,&linesearch->norms,NULL));

  if (linesearch->ops->setfromoptions) PetscCall((*linesearch->ops->setfromoptions)(PetscOptionsObject,linesearch));

  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)linesearch));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchView - Prints useful information about the line search

   Input Parameters:
.  linesearch - linesearch context

   Logically Collective on SNESLineSearch

   Level: intermediate

.seealso: `SNESLineSearchCreate()`
@*/
PetscErrorCode SNESLineSearchView(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (!viewer) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)linesearch),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(linesearch,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)linesearch,viewer));
    if (linesearch->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall((*linesearch->ops->view)(linesearch,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maxstep=%e, minlambda=%e\n", (double)linesearch->maxstep,(double)linesearch->steptol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%e, absolute=%e, lambda=%e\n", (double)linesearch->rtol,(double)linesearch->atol,(double)linesearch->ltol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maximum iterations=%" PetscInt_FMT "\n", linesearch->max_its));
    if (linesearch->ops->precheck) {
      if (linesearch->ops->precheck == SNESLineSearchPreCheckPicard) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  using precheck step to speed up Picard convergence\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  using user-defined precheck step\n"));
      }
    }
    if (linesearch->ops->postcheck) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using user-defined postcheck step\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchGetType - Gets the linesearch type

   Logically Collective on SNESLineSearch

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
-  type - The type of line search, or NULL if not set

   Level: intermediate

.seealso: `SNESLineSearchCreate()`, `SNESLineSearchType`, `SNESLineSearchSetFromOptions()`, `SNESLineSearchSetType()`
@*/
PetscErrorCode SNESLineSearchGetType(SNESLineSearch linesearch, SNESLineSearchType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)linesearch)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetType - Sets the linesearch type

   Logically Collective on SNESLineSearch

   Input Parameters:
+  linesearch - linesearch context
-  type - The type of line search to be used

   Available Types:
+  SNESLINESEARCHBASIC - (or equivalently SNESLINESEARCHNONE) Simple damping line search, defaults to using the full Newton step
.  SNESLINESEARCHBT - Backtracking line search over the L2 norm of the function
.  SNESLINESEARCHL2 - Secant line search over the L2 norm of the function
.  SNESLINESEARCHCP - Critical point secant line search assuming F(x) = grad G(x) for some unknown G(x)
.  SNESLINESEARCHNLEQERR - Affine-covariant error-oriented linesearch
-  SNESLINESEARCHSHELL - User provided SNESLineSearch implementation

   Options Database:
.  -snes_linesearch_type <type> - basic (or equivalently none), bt, l2, cp, nleqerr, shell

   Level: intermediate

.seealso: `SNESLineSearchCreate()`, `SNESLineSearchType`, `SNESLineSearchSetFromOptions()`, `SNESLineSearchGetType()`
@*/
PetscErrorCode SNESLineSearchSetType(SNESLineSearch linesearch, SNESLineSearchType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(SNESLineSearch);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(SNESLineSearchList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested Line Search type %s",type);
  /* Destroy the previous private linesearch context */
  if (linesearch->ops->destroy) {
    PetscCall((*(linesearch)->ops->destroy)(linesearch));
    linesearch->ops->destroy = NULL;
  }
  /* Reinitialize function pointers in SNESLineSearchOps structure */
  linesearch->ops->apply          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->destroy        = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)linesearch,type));
  PetscCall((*r)(linesearch));
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetSNES - Sets the SNES for the linesearch for function evaluation.

   Input Parameters:
+  linesearch - linesearch context
-  snes - The snes instance

   Level: developer

   Notes:
   This happens automatically when the line search is obtained/created with
   SNESGetLineSearch().  This routine is therefore mainly called within SNES
   implementations.

   Level: developer

.seealso: `SNESLineSearchGetSNES()`, `SNESLineSearchSetVecs()`, `SNES`
@*/
PetscErrorCode  SNESLineSearchSetSNES(SNESLineSearch linesearch, SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidHeaderSpecific(snes,SNES_CLASSID,2);
  linesearch->snes = snes;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetSNES - Gets the SNES instance associated with the line search.
   Having an associated SNES is necessary because most line search implementations must be able to
   evaluate the function using SNESComputeFunction() for the associated SNES.  This routine
   is used in the line search implementations when one must get this associated SNES instance.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  snes - The snes instance

   Level: developer

.seealso: `SNESLineSearchGetSNES()`, `SNESLineSearchSetVecs()`, `SNES`
@*/
PetscErrorCode  SNESLineSearchGetSNES(SNESLineSearch linesearch, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(snes,2);
  *snes = linesearch->snes;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetLambda - Gets the last linesearch steplength discovered.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  lambda - The last steplength computed during SNESLineSearchApply()

   Level: advanced

   Notes:
   This is useful in methods where the solver is ill-scaled and
   requires some adaptive notion of the difference in scale between the
   solution and the function.  For instance, SNESQN may be scaled by the
   line search lambda using the argument -snes_qn_scaling ls.

.seealso: `SNESLineSearchSetLambda()`, `SNESLineSearchGetDamping()`, `SNESLineSearchApply()`
@*/
PetscErrorCode  SNESLineSearchGetLambda(SNESLineSearch linesearch,PetscReal *lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidRealPointer(lambda, 2);
  *lambda = linesearch->lambda;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetLambda - Sets the linesearch steplength.

   Input Parameters:
+  linesearch - linesearch context
-  lambda - The last steplength.

   Notes:
   This routine is typically used within implementations of SNESLineSearchApply()
   to set the final steplength.  This routine (and SNESLineSearchGetLambda()) were
   added in order to facilitate Quasi-Newton methods that use the previous steplength
   as an inner scaling parameter.

   Level: advanced

.seealso: `SNESLineSearchGetLambda()`
@*/
PetscErrorCode  SNESLineSearchSetLambda(SNESLineSearch linesearch, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->lambda = lambda;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetTolerances - Gets the tolerances for the linesearch.  These include
   tolerances for the relative and absolute change in the function norm, the change
   in lambda for iterative line searches, the minimum steplength, the maximum steplength,
   and the maximum number of iterations the line search procedure may take.

   Input Parameter:
.  linesearch - linesearch context

   Output Parameters:
+  steptol - The minimum steplength
.  maxstep - The maximum steplength
.  rtol    - The relative tolerance for iterative line searches
.  atol    - The absolute tolerance for iterative line searches
.  ltol    - The change in lambda tolerance for iterative line searches
-  max_it  - The maximum number of iterations of the line search

   Level: intermediate

   Notes:
   Different line searches may implement these parameters slightly differently as
   the type requires.

.seealso: `SNESLineSearchSetTolerances()`
@*/
PetscErrorCode  SNESLineSearchGetTolerances(SNESLineSearch linesearch,PetscReal *steptol,PetscReal *maxstep, PetscReal *rtol, PetscReal *atol, PetscReal *ltol, PetscInt *max_its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (steptol) {
    PetscValidRealPointer(steptol, 2);
    *steptol = linesearch->steptol;
  }
  if (maxstep) {
    PetscValidRealPointer(maxstep, 3);
    *maxstep = linesearch->maxstep;
  }
  if (rtol) {
    PetscValidRealPointer(rtol, 4);
    *rtol = linesearch->rtol;
  }
  if (atol) {
    PetscValidRealPointer(atol, 5);
    *atol = linesearch->atol;
  }
  if (ltol) {
    PetscValidRealPointer(ltol, 6);
    *ltol = linesearch->ltol;
  }
  if (max_its) {
    PetscValidIntPointer(max_its, 7);
    *max_its = linesearch->max_its;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetTolerances -  Gets the tolerances for the linesearch.  These include
   tolerances for the relative and absolute change in the function norm, the change
   in lambda for iterative line searches, the minimum steplength, the maximum steplength,
   and the maximum number of iterations the line search procedure may take.

   Input Parameters:
+  linesearch - linesearch context
.  steptol - The minimum steplength
.  maxstep - The maximum steplength
.  rtol    - The relative tolerance for iterative line searches
.  atol    - The absolute tolerance for iterative line searches
.  ltol    - The change in lambda tolerance for iterative line searches
-  max_it  - The maximum number of iterations of the line search

   Notes:
   The user may choose to not set any of the tolerances using PETSC_DEFAULT in
   place of an argument.

   Level: intermediate

.seealso: `SNESLineSearchGetTolerances()`
@*/
PetscErrorCode  SNESLineSearchSetTolerances(SNESLineSearch linesearch,PetscReal steptol,PetscReal maxstep, PetscReal rtol, PetscReal atol, PetscReal ltol, PetscInt max_its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidLogicalCollectiveReal(linesearch,steptol,2);
  PetscValidLogicalCollectiveReal(linesearch,maxstep,3);
  PetscValidLogicalCollectiveReal(linesearch,rtol,4);
  PetscValidLogicalCollectiveReal(linesearch,atol,5);
  PetscValidLogicalCollectiveReal(linesearch,ltol,6);
  PetscValidLogicalCollectiveInt(linesearch,max_its,7);

  if (steptol!= PETSC_DEFAULT) {
    PetscCheck(steptol >= 0.0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Minimum step length %14.12e must be non-negative",(double)steptol);
    linesearch->steptol = steptol;
  }

  if (maxstep!= PETSC_DEFAULT) {
    PetscCheck(maxstep >= 0.0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Maximum step length %14.12e must be non-negative",(double)maxstep);
    linesearch->maxstep = maxstep;
  }

  if (rtol != PETSC_DEFAULT) {
    PetscCheck(rtol >= 0.0 && rtol < 1.0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %14.12e must be non-negative and less than 1.0",(double)rtol);
    linesearch->rtol = rtol;
  }

  if (atol != PETSC_DEFAULT) {
    PetscCheck(atol >= 0.0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %14.12e must be non-negative",(double)atol);
    linesearch->atol = atol;
  }

  if (ltol != PETSC_DEFAULT) {
    PetscCheck(ltol >= 0.0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Lambda tolerance %14.12e must be non-negative",(double)ltol);
    linesearch->ltol = ltol;
  }

  if (max_its != PETSC_DEFAULT) {
    PetscCheck(max_its >= 0,PetscObjectComm((PetscObject)linesearch),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %" PetscInt_FMT " must be non-negative",max_its);
    linesearch->max_its = max_its;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetDamping - Gets the line search damping parameter.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  damping - The damping parameter

   Level: advanced

.seealso: `SNESLineSearchGetStepTolerance()`, `SNESQN`
@*/

PetscErrorCode  SNESLineSearchGetDamping(SNESLineSearch linesearch,PetscReal *damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidRealPointer(damping, 2);
  *damping = linesearch->damping;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetDamping - Sets the line search damping parameter.

   Input Parameters:
+  linesearch - linesearch context
-  damping - The damping parameter

   Options Database:
.   -snes_linesearch_damping
   Level: intermediate

   Notes:
   The basic (also known as the none) line search merely takes the update step scaled by the damping parameter.
   The use of the damping parameter in the l2 and cp line searches is much more subtle;
   it is used as a starting point in calculating the secant step. However, the eventual
   step may be of greater length than the damping parameter.  In the bt line search it is
   used as the maximum possible step length, as the bt line search only backtracks.

.seealso: `SNESLineSearchGetDamping()`
@*/
PetscErrorCode  SNESLineSearchSetDamping(SNESLineSearch linesearch,PetscReal damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->damping = damping;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetOrder - Gets the line search approximation order.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  order - The order

   Possible Values for order:
+  1 or SNES_LINESEARCH_ORDER_LINEAR - linear order
.  2 or SNES_LINESEARCH_ORDER_QUADRATIC - quadratic order
-  3 or SNES_LINESEARCH_ORDER_CUBIC - cubic order

   Level: intermediate

.seealso: `SNESLineSearchSetOrder()`
@*/

PetscErrorCode  SNESLineSearchGetOrder(SNESLineSearch linesearch,PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidIntPointer(order, 2);
  *order = linesearch->order;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetOrder - Sets the maximum order of the polynomial fit used in the line search

   Input Parameters:
+  linesearch - linesearch context
-  order - The damping parameter

   Level: intermediate

   Possible Values for order:
+  1 or SNES_LINESEARCH_ORDER_LINEAR - linear order
.  2 or SNES_LINESEARCH_ORDER_QUADRATIC - quadratic order
-  3 or SNES_LINESEARCH_ORDER_CUBIC - cubic order

   Notes:
   Variable orders are supported by the following line searches:
+  bt - cubic and quadratic
-  cp - linear and quadratic

.seealso: `SNESLineSearchGetOrder()`, `SNESLineSearchSetDamping()`
@*/
PetscErrorCode  SNESLineSearchSetOrder(SNESLineSearch linesearch,PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->order = order;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetNorms - Gets the norms for for X, Y, and F.

   Input Parameter:
.  linesearch - linesearch context

   Output Parameters:
+  xnorm - The norm of the current solution
.  fnorm - The norm of the current function
-  ynorm - The norm of the current update

   Notes:
   This function is mainly called from SNES implementations.

   Level: developer

.seealso: `SNESLineSearchSetNorms()` `SNESLineSearchGetVecs()`
@*/
PetscErrorCode  SNESLineSearchGetNorms(SNESLineSearch linesearch, PetscReal * xnorm, PetscReal * fnorm, PetscReal * ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (xnorm) *xnorm = linesearch->xnorm;
  if (fnorm) *fnorm = linesearch->fnorm;
  if (ynorm) *ynorm = linesearch->ynorm;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetNorms - Gets the computed norms for for X, Y, and F.

   Input Parameters:
+  linesearch - linesearch context
.  xnorm - The norm of the current solution
.  fnorm - The norm of the current function
-  ynorm - The norm of the current update

   Level: advanced

.seealso: `SNESLineSearchGetNorms()`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode  SNESLineSearchSetNorms(SNESLineSearch linesearch, PetscReal xnorm, PetscReal fnorm, PetscReal ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->xnorm = xnorm;
  linesearch->fnorm = fnorm;
  linesearch->ynorm = ynorm;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchComputeNorms - Computes the norms of X, F, and Y.

   Input Parameters:
.  linesearch - linesearch context

   Options Database Keys:
.   -snes_linesearch_norms - turn norm computation on or off

   Level: intermediate

.seealso: `SNESLineSearchGetNorms`, `SNESLineSearchSetNorms()`, `SNESLineSearchSetComputeNorms()`
@*/
PetscErrorCode SNESLineSearchComputeNorms(SNESLineSearch linesearch)
{
  SNES           snes;

  PetscFunctionBegin;
  if (linesearch->norms) {
    if (linesearch->ops->vinorm) {
      PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
      PetscCall(VecNorm(linesearch->vec_sol, NORM_2, &linesearch->xnorm));
      PetscCall(VecNorm(linesearch->vec_update, NORM_2, &linesearch->ynorm));
      PetscCall((*linesearch->ops->vinorm)(snes, linesearch->vec_func, linesearch->vec_sol, &linesearch->fnorm));
    } else {
      PetscCall(VecNormBegin(linesearch->vec_func,   NORM_2, &linesearch->fnorm));
      PetscCall(VecNormBegin(linesearch->vec_sol,    NORM_2, &linesearch->xnorm));
      PetscCall(VecNormBegin(linesearch->vec_update, NORM_2, &linesearch->ynorm));
      PetscCall(VecNormEnd(linesearch->vec_func,     NORM_2, &linesearch->fnorm));
      PetscCall(VecNormEnd(linesearch->vec_sol,      NORM_2, &linesearch->xnorm));
      PetscCall(VecNormEnd(linesearch->vec_update,   NORM_2, &linesearch->ynorm));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetComputeNorms - Turns on or off the computation of final norms in the line search.

   Input Parameters:
+  linesearch  - linesearch context
-  flg  - indicates whether or not to compute norms

   Options Database Keys:
.   -snes_linesearch_norms <true> - Turns on/off computation of the norms for basic (none) linesearch

   Notes:
   This is most relevant to the SNESLINESEARCHBASIC (or equivalently SNESLINESEARCHNONE) line search type since most line searches have a stopping criteria involving the norm.

   Level: intermediate

.seealso: `SNESLineSearchGetNorms()`, `SNESLineSearchSetNorms()`, `SNESLineSearchComputeNorms()`, `SNESLINESEARCHBASIC`
@*/
PetscErrorCode SNESLineSearchSetComputeNorms(SNESLineSearch linesearch, PetscBool flg)
{
  PetscFunctionBegin;
  linesearch->norms = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetVecs - Gets the vectors from the SNESLineSearch context

   Input Parameter:
.  linesearch - linesearch context

   Output Parameters:
+  X - Solution vector
.  F - Function vector
.  Y - Search direction vector
.  W - Solution work vector
-  G - Function work vector

   Notes:
   At the beginning of a line search application, X should contain a
   solution and the vector F the function computed at X.  At the end of the
   line search application, X should contain the new solution, and F the
   function evaluated at the new solution.

   These vectors are owned by the SNESLineSearch and should not be destroyed by the caller

   Level: advanced

.seealso: `SNESLineSearchGetNorms()`, `SNESLineSearchSetVecs()`
@*/
PetscErrorCode SNESLineSearchGetVecs(SNESLineSearch linesearch,Vec *X,Vec *F, Vec *Y,Vec *W,Vec *G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (X) {
    PetscValidPointer(X, 2);
    *X = linesearch->vec_sol;
  }
  if (F) {
    PetscValidPointer(F, 3);
    *F = linesearch->vec_func;
  }
  if (Y) {
    PetscValidPointer(Y, 4);
    *Y = linesearch->vec_update;
  }
  if (W) {
    PetscValidPointer(W, 5);
    *W = linesearch->vec_sol_new;
  }
  if (G) {
    PetscValidPointer(G, 6);
    *G = linesearch->vec_func_new;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetVecs - Sets the vectors on the SNESLineSearch context

   Input Parameters:
+  linesearch - linesearch context
.  X - Solution vector
.  F - Function vector
.  Y - Search direction vector
.  W - Solution work vector
-  G - Function work vector

   Level: advanced

.seealso: `SNESLineSearchSetNorms()`, `SNESLineSearchGetVecs()`
@*/
PetscErrorCode SNESLineSearchSetVecs(SNESLineSearch linesearch,Vec X,Vec F,Vec Y,Vec W, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
    linesearch->vec_sol = X;
  }
  if (F) {
    PetscValidHeaderSpecific(F,VEC_CLASSID,3);
    linesearch->vec_func = F;
  }
  if (Y) {
    PetscValidHeaderSpecific(Y,VEC_CLASSID,4);
    linesearch->vec_update = Y;
  }
  if (W) {
    PetscValidHeaderSpecific(W,VEC_CLASSID,5);
    linesearch->vec_sol_new = W;
  }
  if (G) {
    PetscValidHeaderSpecific(G,VEC_CLASSID,6);
    linesearch->vec_func_new = G;
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchAppendOptionsPrefix - Appends to the prefix used for searching for all
   SNES options in the database.

   Logically Collective on SNESLineSearch

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: `SNESGetOptionsPrefix()`
@*/
PetscErrorCode  SNESLineSearchAppendOptionsPrefix(SNESLineSearch linesearch,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)linesearch,prefix));
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchGetOptionsPrefix - Sets the prefix used for searching for all
   SNESLineSearch options in the database.

   Not Collective

   Input Parameter:
.  linesearch - the SNESLineSearch context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
   On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: `SNESAppendOptionsPrefix()`
@*/
PetscErrorCode  SNESLineSearchGetOptionsPrefix(SNESLineSearch linesearch,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)linesearch,prefix));
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetWorkVecs - Gets work vectors for the line search.

   Input Parameters:
+  linesearch - the SNESLineSearch context
-  nwork - the number of work vectors

   Level: developer

.seealso: `SNESSetWorkVecs()`
@*/
PetscErrorCode  SNESLineSearchSetWorkVecs(SNESLineSearch linesearch, PetscInt nwork)
{
  PetscFunctionBegin;
  if (linesearch->vec_sol) {
    PetscCall(VecDuplicateVecs(linesearch->vec_sol, nwork, &linesearch->work));
  } else SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "Cannot get linesearch work-vectors without setting a solution vec!");
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchGetReason - Gets the success/failure status of the last line search application

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  result - The success or failure status

   Notes:
   This is typically called after SNESLineSearchApply() in order to determine if the line-search failed
   (and set the SNES convergence accordingly).

   Level: intermediate

.seealso: `SNESLineSearchSetReason()`, `SNESLineSearchReason`
@*/
PetscErrorCode  SNESLineSearchGetReason(SNESLineSearch linesearch, SNESLineSearchReason *result)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  PetscValidPointer(result, 2);
  *result = linesearch->result;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchSetReason - Sets the success/failure status of the last line search application

   Input Parameters:
+  linesearch - linesearch context
-  result - The success or failure status

   Notes:
   This is typically called in a SNESLineSearchApply() or SNESLineSearchShell implementation to set
   the success or failure of the line search method.

   Level: developer

.seealso: `SNESLineSearchGetSResult()`
@*/
PetscErrorCode  SNESLineSearchSetReason(SNESLineSearch linesearch, SNESLineSearchReason result)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  linesearch->result = result;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchSetVIFunctions - Sets VI-specific functions for line search computation.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  projectfunc - function for projecting the function to the bounds
-  normfunc - function for computing the norm of an active set

   Logically Collective on SNES

   Calling sequence of projectfunc:
.vb
   projectfunc (SNES snes, Vec X)
.ve

    Input parameters for projectfunc:
+   snes - nonlinear context
-   X - current solution

    Output parameters for projectfunc:
.   X - Projected solution

   Calling sequence of normfunc:
.vb
   projectfunc (SNES snes, Vec X, Vec F, PetscScalar * fnorm)
.ve

    Input parameters for normfunc:
+   snes - nonlinear context
.   X - current solution
-   F - current residual

    Output parameters for normfunc:
.   fnorm - VI-specific norm of the function

    Notes:
    The VI solvers require projection of the solution to the feasible set.  projectfunc should implement this.

    The VI solvers require special evaluation of the function norm such that the norm is only calculated
    on the inactive set.  This should be implemented by normfunc.

    Level: developer

.seealso: `SNESLineSearchGetVIFunctions()`, `SNESLineSearchSetPostCheck()`, `SNESLineSearchSetPreCheck()`
@*/
PetscErrorCode SNESLineSearchSetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFunc projectfunc, SNESLineSearchVINormFunc normfunc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  if (projectfunc) linesearch->ops->viproject = projectfunc;
  if (normfunc) linesearch->ops->vinorm = normfunc;
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchGetVIFunctions - Sets VI-specific functions for line search computation.

   Input Parameter:
.  linesearch - the line search context, obtain with SNESGetLineSearch()

   Output Parameters:
+  projectfunc - function for projecting the function to the bounds
-  normfunc - function for computing the norm of an active set

   Logically Collective on SNES

    Level: developer

.seealso: `SNESLineSearchSetVIFunctions()`, `SNESLineSearchGetPostCheck()`, `SNESLineSearchGetPreCheck()`
@*/
PetscErrorCode SNESLineSearchGetVIFunctions(SNESLineSearch linesearch, SNESLineSearchVIProjectFunc *projectfunc, SNESLineSearchVINormFunc *normfunc)
{
  PetscFunctionBegin;
  if (projectfunc) *projectfunc = linesearch->ops->viproject;
  if (normfunc) *normfunc = linesearch->ops->vinorm;
  PetscFunctionReturn(0);
}

/*@C
  SNESLineSearchRegister - See SNESLineSearchRegister()

  Level: advanced
@*/
PetscErrorCode  SNESLineSearchRegister(const char sname[],PetscErrorCode (*function)(SNESLineSearch))
{
  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscCall(PetscFunctionListAdd(&SNESLineSearchList,sname,function));
  PetscFunctionReturn(0);
}
