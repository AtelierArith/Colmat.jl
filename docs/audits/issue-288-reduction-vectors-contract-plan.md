# Issue #288: ORGBR/ORMBR and ORGHR/ORMHR C contracts

Status: the 32 S/D/C/Z high/work entries pass focused layout, diagnostic and
independent-call checks. The immutable full suite is recorded in
`docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{orgbr,ungbr,ormbr,unmbr,orghr,unghr,ormhr,unmhr}` and their
`_work` forms (32 entries). They use the existing native bidiagonal/Hessenberg
Q generators and applicators.

## Contracts

- ORGBR/UNGBR select VECT=`Q` or `P`, with `0 <= K <= N` for Q and `0 <= K <= M`
  for P. Only the logical M-by-N A is converted. Row-major requires LDA >= N
  (-7).
- ORGHR/UNGHR generate Q from the Hessenberg reduction; `ILO`/`IHI` follow the
  reference range. Row-major requires LDA >= N (-6).
- ORMBR/UNMBR apply Q or P with SIDE/TRANS. The reflector order is
  `NQ = M` for SIDE='L' and `NQ = N` for SIDE='R'; A is `NQ`-by-`MIN(NQ,K)` for
  VECT='Q' and the transpose for VECT='P'. Row-major requires LDA >= the
  logical A column count (-9) and LDC >= N (-12).
- ORMHR/UNMHR apply the Hessenberg Q with R = M or N. Row-major requires
  LDA >= R (-9) and LDC >= N (-12).
- High-level NaN order: ORGBR A (-6), tau (-8); ORGHR A (-5), tau (-7);
  ORMBR A (-8), C (-11), tau (-10); ORMHR A (-8), C (-11), tau (-10).
- High-level entries query LWORK=-1, allocate the requested workspace and call
  the work entry. Workspace errors and invalid flags use the reference indices.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, VECT Q/P, SIDE L/R, TRANS N/T/C, rectangular C, row/column-major
  equivalence, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
