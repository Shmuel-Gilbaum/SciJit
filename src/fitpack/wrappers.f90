! C-callable bind(c) wrappers for the FITPACK evaluator routines.
! Companion to test_fit.f90 (which wraps bispev/bispeu); this module adds
! the remaining "evaluator group": univariate evaluation/integration/roots,
! parametric curve evaluation, and bivariate derivative/integral/profile
! routines. Signatures mirror fitpack/functions.md exactly.
module fitpack_wrappers
  use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
  implicit none
  private
  public  splev_wrapper, splder_wrapper, splint_wrapper, spalde_wrapper, &
          sproot_wrapper, curev_wrapper, surev_wrapper, cualde_wrapper, &
          parder_wrapper, pardeu_wrapper, dblint_wrapper, fourco_wrapper, &
          insert_wrapper, profil_wrapper, &
          curfit_wrapper, percur_wrapper, parcur_wrapper, clocur_wrapper, &
          concur_wrapper, cocosp_wrapper, concon_wrapper, regrid_wrapper, &
          parsur_wrapper, pogrid_wrapper, spgrid_wrapper, sphere_wrapper, &
          surfit_wrapper, polar_wrapper, evapol_wrapper, &
          bispev_wrapper, bispeu_wrapper

  ! interface of the user-supplied boundary function rad(v) needed by
  ! polar/evapol: plain Fortran function, argument passed by reference,
  ! i.e. the C callback receives double* and returns double.
  abstract interface
    function rad_iface(v) result(r)
      import :: c_double
      real(c_double), intent(in) :: v
      real(c_double) :: r
    end function
  end interface
contains

  ! ---------------- univariate splines ----------------

  subroutine splev_wrapper(t,n,c,k,x,y,m,e,ier) bind(c)
    integer(c_int), intent(in) :: n, k, m, e
    real(c_double), intent(in) :: t(n), c(n), x(m)
    real(c_double), intent(inout) :: y(m)
    integer(c_int), intent(out) :: ier

    call splev(t,n,c,k,x,y,m,e,ier)

  end subroutine

  subroutine splder_wrapper(t,n,c,k,nu,x,y,m,e,wrk,ier) bind(c)
    integer(c_int), intent(in) :: n, k, nu, m, e
    real(c_double), intent(in) :: t(n), c(n), x(m)
    real(c_double), intent(inout) :: y(m), wrk(n)
    integer(c_int), intent(out) :: ier

    call splder(t,n,c,k,nu,x,y,m,e,wrk,ier)

  end subroutine

  subroutine splint_wrapper(t,n,c,k,a,b,wrk,res) bind(c)
    integer(c_int), intent(in) :: n, k
    real(c_double), intent(in) :: t(n), c(n), a, b
    real(c_double), intent(inout) :: wrk(n)
    real(c_double), intent(out) :: res
    real(c_double), external :: splint

    res = splint(t,n,c,k,a,b,wrk)

  end subroutine

  subroutine spalde_wrapper(t,n,c,k1,x,d,ier) bind(c)
    integer(c_int), intent(in) :: n, k1
    real(c_double), intent(in) :: t(n), c(n), x
    real(c_double), intent(inout) :: d(k1)
    integer(c_int), intent(out) :: ier

    call spalde(t,n,c,k1,x,d,ier)

  end subroutine

  subroutine sproot_wrapper(t,n,c,zero,mest,m,ier) bind(c)
    integer(c_int), intent(in) :: n, mest
    real(c_double), intent(in) :: t(n), c(n)
    real(c_double), intent(inout) :: zero(mest)
    integer(c_int), intent(out) :: m, ier

    call sproot(t,n,c,zero,mest,m,ier)

  end subroutine

  subroutine fourco_wrapper(t,n,c,alfa,m,ress,resc,wrk1,wrk2,ier) bind(c)
    integer(c_int), intent(in) :: n, m
    real(c_double), intent(in) :: t(n), c(n), alfa(m)
    real(c_double), intent(inout) :: ress(m), resc(m), wrk1(n), wrk2(n)
    integer(c_int), intent(out) :: ier

    call fourco(t,n,c,alfa,m,ress,resc,wrk1,wrk2,ier)

  end subroutine

  subroutine insert_wrapper(iopt,t,n,c,k,x,tt,nn,cc,nest,ier) bind(c)
    integer(c_int), intent(in) :: iopt, n, k, nest
    real(c_double), intent(in) :: t(nest), c(nest), x
    real(c_double), intent(inout) :: tt(nest), cc(nest)
    integer(c_int), intent(out) :: nn, ier

    call insert(iopt,t,n,c,k,x,tt,nn,cc,nest,ier)

  end subroutine

  ! ---------------- parametric curves ----------------

  subroutine curev_wrapper(idim,t,n,c,nc,k,u,m,x,mx,ier) bind(c)
    integer(c_int), intent(in) :: idim, n, nc, k, m, mx
    real(c_double), intent(in) :: t(n), c(nc), u(m)
    real(c_double), intent(inout) :: x(mx)
    integer(c_int), intent(out) :: ier

    call curev(idim,t,n,c,nc,k,u,m,x,mx,ier)

  end subroutine

  subroutine cualde_wrapper(idim,t,n,c,nc,k1,u,d,nd,ier) bind(c)
    integer(c_int), intent(in) :: idim, n, nc, k1, nd
    real(c_double), intent(in) :: t(n), c(nc), u
    real(c_double), intent(inout) :: d(nd)
    integer(c_int), intent(out) :: ier

    call cualde(idim,t,n,c,nc,k1,u,d,nd,ier)

  end subroutine

  ! ---------------- bivariate / surface routines ----------------

  subroutine parder_wrapper(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: nx, ny, kx, ky, nux, nuy, mx, my, lwrk, kwrk
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1))
    real(c_double), intent(in) :: x(mx), y(my)
    real(c_double), intent(inout) :: z(mx*my), wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call parder(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier)

  end subroutine

  subroutine pardeu_wrapper(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,y,z,m,wrk,lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: nx, ny, kx, ky, nux, nuy, m, lwrk, kwrk
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1))
    real(c_double), intent(in) :: x(m), y(m)
    real(c_double), intent(inout) :: z(m), wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call pardeu(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,y,z,m,wrk,lwrk,iwrk,kwrk,ier)

  end subroutine

  subroutine dblint_wrapper(tx,nx,ty,ny,c,kx,ky,xb,xe,yb,ye,wrk,res) bind(c)
    integer(c_int), intent(in) :: nx, ny, kx, ky
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1))
    real(c_double), intent(in) :: xb, xe, yb, ye
    real(c_double), intent(inout) :: wrk(nx+ny-kx-ky-2)
    real(c_double), intent(out) :: res
    real(c_double), external :: dblint

    res = dblint(tx,nx,ty,ny,c,kx,ky,xb,xe,yb,ye,wrk)

  end subroutine

  subroutine profil_wrapper(iopt,tx,nx,ty,ny,c,kx,ky,u,nu,cu,ier) bind(c)
    integer(c_int), intent(in) :: iopt, nx, ny, kx, ky, nu
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1)), u
    real(c_double), intent(inout) :: cu(nu)
    integer(c_int), intent(out) :: ier

    call profil(iopt,tx,nx,ty,ny,c,kx,ky,u,nu,cu,ier)

  end subroutine

  subroutine surev_wrapper(idim,tu,nu,tv,nv,c,u,mu,v,mv,f,mf,wrk,lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: idim, nu, nv, mu, mv, mf, lwrk, kwrk
    real(c_double), intent(in) :: tu(nu), tv(nv), c((nu-4)*(nv-4)*idim)
    real(c_double), intent(in) :: u(mu), v(mv)
    real(c_double), intent(inout) :: f(mf), wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call surev(idim,tu,nu,tv,nv,c,u,mu,v,mv,f,mf,wrk,lwrk,iwrk,kwrk,ier)

  end subroutine

  ! ---------------- curve fitting ----------------

  subroutine curfit_wrapper(iopt,m,x,y,w,xb,xe,k,s,nest,n,t,c,fp, &
                            wrk,lwrk,iwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, m, k, nest, lwrk
    real(c_double), intent(in) :: x(m), y(m), w(m), xb, xe, s
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nest), fp, wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(nest)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call curfit(iopt,m,x,y,w,xb,xe,k,s,nest,n,t,c,fp,wrk,lwrk,iwrk,tol,maxit,ier)

  end subroutine

  subroutine percur_wrapper(iopt,m,x,y,w,k,s,nest,n,t,c,fp, &
                            wrk,lwrk,iwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, m, k, nest, lwrk
    real(c_double), intent(in) :: x(m), y(m), w(m), s
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nest), fp, wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(nest)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call percur(iopt,m,x,y,w,k,s,nest,n,t,c,fp,wrk,lwrk,iwrk,tol,maxit,ier)

  end subroutine

  subroutine parcur_wrapper(iopt,ipar,idim,m,u,mx,x,w,ub,ue,k,s,nest,n,t, &
                            nc,c,fp,wrk,lwrk,iwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, ipar, idim, m, mx, k, nest, nc, lwrk
    real(c_double), intent(in) :: x(mx), w(m), s
    real(c_double), intent(inout) :: u(m), ub, ue
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nc), fp, wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(nest)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call parcur(iopt,ipar,idim,m,u,mx,x,w,ub,ue,k,s,nest,n,t,nc,c,fp, &
                wrk,lwrk,iwrk,tol,maxit,ier)

  end subroutine

  subroutine clocur_wrapper(iopt,ipar,idim,m,u,mx,x,w,k,s,nest,n,t,nc,c,fp, &
                            wrk,lwrk,iwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, ipar, idim, m, mx, k, nest, nc, lwrk
    real(c_double), intent(in) :: x(mx), w(m), s
    real(c_double), intent(inout) :: u(m)
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nc), fp, wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(nest)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call clocur(iopt,ipar,idim,m,u,mx,x,w,k,s,nest,n,t,nc,c,fp, &
                wrk,lwrk,iwrk,tol,maxit,ier)

  end subroutine

  subroutine concur_wrapper(iopt,idim,m,u,mx,x,xx,w,ib,db,nb,ie,de,ne,k,s, &
                            nest,n,t,nc,c,np,cp,fp,wrk,lwrk,iwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, idim, m, mx, ib, nb, ie, ne, k, &
                                  nest, nc, np, lwrk
    real(c_double), intent(in) :: u(m), x(mx), w(m), s
    real(c_double), intent(inout) :: xx(mx), db(nb), de(ne)
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nc), cp(np), fp, wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(nest)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call concur(iopt,idim,m,u,mx,x,xx,w,ib,db,nb,ie,de,ne,k,s,nest,n,t, &
                nc,c,np,cp,fp,wrk,lwrk,iwrk,tol,maxit,ier)

  end subroutine

  ! bind/logical arrays are received as integer(c_int) (gfortran logical*4
  ! and integer*4 share size and 0/1 representation; fitpack only ever
  ! sets/reads .true./.false. through them via implicit interfaces).
  subroutine cocosp_wrapper(m,x,y,w,n,t,e,maxtr,maxbin,c,sq,sx,bnd,wrk, &
                            lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: m, n, maxtr, maxbin, lwrk, kwrk
    real(c_double), intent(in) :: x(m), y(m), w(m), t(n)
    real(c_double), intent(inout) :: e(n), c(n), sq, sx(m), wrk(lwrk)
    integer(c_int), intent(inout) :: bnd(n), iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call cocosp(m,x,y,w,n,t,e,maxtr,maxbin,c,sq,sx,bnd,wrk,lwrk, &
                iwrk,kwrk,ier)

  end subroutine

  subroutine concon_wrapper(iopt,m,x,y,w,v,s,nest,maxtr,maxbin,n,t,c,sq, &
                            sx,bnd,wrk,lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: iopt, m, nest, maxtr, maxbin, lwrk, kwrk
    real(c_double), intent(in) :: x(m), y(m), w(m), s
    real(c_double), intent(inout) :: v(m)
    integer(c_int), intent(inout) :: n
    real(c_double), intent(inout) :: t(nest), c(nest), sq, sx(m), wrk(lwrk)
    integer(c_int), intent(inout) :: bnd(nest), iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call concon(iopt,m,x,y,w,v,s,nest,maxtr,maxbin,n,t,c,sq,sx,bnd, &
                wrk,lwrk,iwrk,kwrk,ier)

  end subroutine

  ! ---------------- surface fitting ----------------

  subroutine regrid_wrapper(iopt,mx,x,my,y,z,xb,xe,yb,ye,kx,ky,s, &
                            nxest,nyest,nx,tx,ny,ty,c,fp,wrk,lwrk, &
                            iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, mx, my, kx, ky, nxest, nyest, &
                                  lwrk, kwrk
    real(c_double), intent(in) :: x(mx), y(my), z(mx*my), xb, xe, yb, ye, s
    integer(c_int), intent(inout) :: nx, ny
    real(c_double), intent(inout) :: tx(nxest), ty(nyest)
    real(c_double), intent(inout) :: c((nxest-kx-1)*(nyest-ky-1)), fp
    real(c_double), intent(inout) :: wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call regrid(iopt,mx,x,my,y,z,xb,xe,yb,ye,kx,ky,s,nxest,nyest, &
                nx,tx,ny,ty,c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine parsur_wrapper(iopt,ipar,idim,mu,u,mv,v,f,s,nuest,nvest, &
                            nu,tu,nv,tv,c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, ipar(2), idim, mu, mv, nuest, &
                                  nvest, lwrk, kwrk
    real(c_double), intent(in) :: u(mu), v(mv), f(mu*mv*idim), s
    integer(c_int), intent(inout) :: nu, nv
    real(c_double), intent(inout) :: tu(nuest), tv(nvest)
    real(c_double), intent(inout) :: c((nuest-4)*(nvest-4)*idim), fp
    real(c_double), intent(inout) :: wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call parsur(iopt,ipar,idim,mu,u,mv,v,f,s,nuest,nvest,nu,tu,nv,tv, &
                c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine pogrid_wrapper(iopt,ider,mu,u,mv,v,z,z0,r,s,nuest,nvest, &
                            nu,tu,nv,tv,c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt(3), ider(2), mu, mv, nuest, nvest, &
                                  lwrk, kwrk
    real(c_double), intent(in) :: u(mu), v(mv), z(mu*mv), z0, r, s
    integer(c_int), intent(inout) :: nu, nv
    real(c_double), intent(inout) :: tu(nuest), tv(nvest)
    real(c_double), intent(inout) :: c((nuest-4)*(nvest-4)), fp
    real(c_double), intent(inout) :: wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call pogrid(iopt,ider,mu,u,mv,v,z,z0,r,s,nuest,nvest,nu,tu,nv,tv, &
                c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine spgrid_wrapper(iopt,ider,mu,u,mv,v,r,r0,r1,s,nuest,nvest, &
                            nu,tu,nv,tv,c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt(3), ider(4), mu, mv, nuest, nvest, &
                                  lwrk, kwrk
    real(c_double), intent(in) :: u(mu), v(mv), r(mu*mv), r0, r1, s
    integer(c_int), intent(inout) :: nu, nv
    real(c_double), intent(inout) :: tu(nuest), tv(nvest)
    real(c_double), intent(inout) :: c((nuest-4)*(nvest-4)), fp
    real(c_double), intent(inout) :: wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call spgrid(iopt,ider,mu,u,mv,v,r,r0,r1,s,nuest,nvest,nu,tu,nv,tv, &
                c,fp,wrk,lwrk,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine sphere_wrapper(iopt,m,teta,phi,r,w,s,ntest,npest,eps, &
                            nt,tt,np,tp,c,fp,wrk1,lwrk1,wrk2,lwrk2, &
                            iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, m, ntest, npest, lwrk1, lwrk2, kwrk
    real(c_double), intent(in) :: teta(m), phi(m), r(m), w(m), s, eps
    integer(c_int), intent(inout) :: nt, np
    real(c_double), intent(inout) :: tt(ntest), tp(npest)
    real(c_double), intent(inout) :: c((ntest-4)*(npest-4)), fp
    real(c_double), intent(inout) :: wrk1(lwrk1), wrk2(lwrk2)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call sphere(iopt,m,teta,phi,r,w,s,ntest,npest,eps,nt,tt,np,tp,c,fp, &
                wrk1,lwrk1,wrk2,lwrk2,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine surfit_wrapper(iopt,m,x,y,z,w,xb,xe,yb,ye,kx,ky,s, &
                            nxest,nyest,nmax,eps,nx,tx,ny,ty,c,fp, &
                            wrk1,lwrk1,wrk2,lwrk2,iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt, m, kx, ky, nxest, nyest, nmax, &
                                  lwrk1, lwrk2, kwrk
    real(c_double), intent(in) :: x(m), y(m), z(m), w(m), xb, xe, yb, ye, &
                                  s, eps
    integer(c_int), intent(inout) :: nx, ny
    real(c_double), intent(inout) :: tx(nmax), ty(nmax)
    real(c_double), intent(inout) :: c((nxest-kx-1)*(nyest-ky-1)), fp
    real(c_double), intent(inout) :: wrk1(lwrk1), wrk2(lwrk2)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit

    call surfit(iopt,m,x,y,z,w,xb,xe,yb,ye,kx,ky,s,nxest,nyest,nmax,eps, &
                nx,tx,ny,ty,c,fp,wrk1,lwrk1,wrk2,lwrk2,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  ! ---------------- polar (user-supplied boundary callback) ----------------

  subroutine polar_wrapper(iopt,m,x,y,z,w,rad,s,nuest,nvest,eps,nu,tu, &
                           nv,tv,u,v,c,fp,wrk1,lwrk1,wrk2,lwrk2, &
                           iwrk,kwrk,tol,maxit,ier) bind(c)
    integer(c_int), intent(in) :: iopt(3), m, nuest, nvest, lwrk1, lwrk2, kwrk
    type(c_funptr), intent(in), value :: rad
    real(c_double), intent(in) :: x(m), y(m), z(m), w(m), s, eps
    integer(c_int), intent(inout) :: nu, nv
    real(c_double), intent(inout) :: tu(nuest), tv(nvest), u(m), v(m)
    real(c_double), intent(inout) :: c((nuest-4)*(nvest-4)), fp
    real(c_double), intent(inout) :: wrk1(lwrk1), wrk2(lwrk2)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier
    real(c_double), intent(in) :: tol
    integer(c_int), intent(in) :: maxit
    procedure(rad_iface), pointer :: radp

    call c_f_procpointer(rad, radp)
    call polar(iopt,m,x,y,z,w,radp,s,nuest,nvest,eps,nu,tu,nv,tv,u,v, &
               c,fp,wrk1,lwrk1,wrk2,lwrk2,iwrk,kwrk,tol,maxit,ier)

  end subroutine

  subroutine evapol_wrapper(tu,nu,tv,nv,c,rad,x,y,res) bind(c)
    integer(c_int), intent(in) :: nu, nv
    real(c_double), intent(in) :: tu(nu), tv(nv), c((nu-4)*(nv-4)), x, y
    type(c_funptr), intent(in), value :: rad
    real(c_double), intent(out) :: res
    real(c_double), external :: evapol
    procedure(rad_iface), pointer :: radp

    call c_f_procpointer(rad, radp)
    res = evapol(tu,nu,tv,nv,c,radp,x,y)

  end subroutine

  ! ---------------- bivariate evaluation (used by driver.py) ----------------

  subroutine bispev_wrapper(tx,nx,ty,ny,c,kx,ky,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier) bind(c)
    integer(c_int), intent(in) :: nx, ny, kx, ky, mx, my, lwrk, kwrk
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1))
    real(c_double), intent(in) :: x(mx), y(my)
    real(c_double), intent(inout) :: z(mx*my), wrk(lwrk)
    integer(c_int), intent(inout) :: iwrk(kwrk)
    integer(c_int), intent(out) :: ier

    call bispev(tx,nx,ty,ny,c,kx,ky,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier)

  end subroutine

  subroutine bispeu_wrapper(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier) bind(c)
    integer(c_int), intent(in) :: nx, ny, kx, ky, m, lwrk
    real(c_double), intent(in) :: tx(nx), ty(ny), c((nx-kx-1)*(ny-ky-1))
    real(c_double), intent(in) :: x(m), y(m)
    real(c_double), intent(inout) :: z(m), wrk(lwrk)
    integer(c_int), intent(out) :: ier

    call bispeu(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier)

  end subroutine

end module
