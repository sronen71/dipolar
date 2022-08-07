import sys

import slepc4py
from slepc4py import SLEPc

slepc4py.init(sys.argv)  # it needs to come before petsc import to get options correctly
from petsc4py import PETSc

Print = PETSc.Sys.Print


def solve_eigensystem(A, k=1, x0=None, problem_type="NHEP", which="SM"):
    # Create the result vectors
    xr, xi = A.createVecs()
    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A, None)
    if k <= 100:
        E.setDimensions(nev=k, ncv=PETSc.DECIDE)
    else:
        E.setDimensions(nev=k, mpd=100)
    if problem_type == "HEP":
        ptype = SLEPc.EPS.ProblemType.HEP
    else:
        ptype = SLEPc.EPS.ProblemType.NHEP
    E.setProblemType(ptype)
    if which == "SR":
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    else:
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    if x0 is not None:
        x0_vec = PETSc.Vec().createWithArray(x0)
        E.setInitialSpace([x0_vec])
        # E.setDeflationSpace([x0_vec])
    E.setFromOptions()

    # Solve the eigensystem
    E.solve()
    Print("")
    its = E.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)
    w = []
    v = []
    if nconv > 0:
        Print("")
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            error = E.computeError(i)
            if k.imag != 0.0:
                Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f       %12g" % (k.real, error))
            if k.imag == 0.0:
                w.append(k.real)
            else:
                w.append(k)
            v.append(xr.getArray() + 1j * xi.getArray())
        Print("")

    return w, v
