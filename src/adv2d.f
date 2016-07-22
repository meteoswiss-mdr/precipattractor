

      subroutine advxy (u,v,z,pos,imax,jmax)
c horizontal advection subroutine.
c increase of the advection grid (ag) by one grid point 
c around the boundaries of the model grid (mg):
c          444444444444444
c         10000000000000003
c         10000000000000003
c         10000000000000003
c         10000000000000003
c         10000000000000003
c         10000000000000003
c          222222222222222 
c 0: values of mg. 1, 2, 3, 4: boundary values introduced in ag.
c boundary values of ag are set equal to neighboring boundary
c values of mg if outflow occurs. in this case the boundary values
c of ag are only used for calculating the polynomials.
c if influx occurs into mg, then boundary values of ag are
c taken as z1(j,k), z3(j,k), j=1,jmax and z2(i,k), z4(i,k), i=1,imax.
c z1, z2, z3, z4 have to specified, e.g. z1,z2,z3,z4=0 for no influx
c or prescribed values if mg is nested within a larger model domain.
c velocity field in grid box (i,j) of ag (Courant numbers):
c
c ..............cv(i,j)..............
c :                                 :
c :                                 :
c cu(i-1,j)     zz(i,j)       cu(i,j)
c :                                 :
c :                                 :
c .............cv(i,j-1).............
c

Cf2py intent(in,out) z
Cf2py intent(in) v
Cf2py intent(in) u
Cf2py intent(in) pos
        
      real u(imax,jmax),v(imax,jmax),z(imax,jmax)
      real cu(imax+2,jmax+2),cv(imax+2,jmax+2),zz(imax+2,jmax+2)
      real z1(jmax),z2(imax),z3(jmax),z4(imax)
      logical pos
      integer i,j,nx,ny
c size of ag is (nx,ny): nx=imax+2, ny=jmax+2.    
      nx=imax+2
      ny=jmax+2

c     ---Init arrays with ones
      do j=1,jmax
        z1(j)=0
        z3(j)=0
      enddo
      do i=1,imax
        z2(i)=0
        z4(i)=0
      enddo    
      
c Courant numbers cu, cv in ag:
c linear interpolation of horizontal wind field from the center of
c grid box (i,j) in mg to the boundaries of grid box (i+1,j+1) in ag.
      do j=1,jmax
         cu(1,j+1)=u(1,j)
         do i=1,imax-1
            cu(i+1,j+1)=0.5*(u(i,j)+u(i+1,j))
         enddo
         cu(imax+1,j+1)=u(imax,j) 
         cu(imax+2,j+1)=cu(imax+1,j+1)
      enddo
      do i=1,imax
         cv(i+1,1)=v(i,1)
         do j=1,jmax-1
            cv(i+1,j+1)=0.5*(v(i,j)+v(i,j+1))
         enddo
         cv(i+1,jmax+1)=v(i,jmax) 
         cv(i+1,jmax+2)=cv(i+1,jmax+1)
      enddo
c transport quantity zz in ag: 
      do i=1,imax
      do j=1,jmax
         zz(i+1,j+1)=z(i,j)
      enddo
      enddo
c boundary values of ag:
c left (i=1) and right (i=nx) boundary of ag:
      do j=1,jmax
         if (cu(1,j+1).ge.0.) then
            zz(1,j+1)=z1(j)
         else
            zz(1,j+1)=z(1,j)
         endif
         if (cu(nx-1,j+1).le.0.) then
            zz(nx,j+1)=z3(j)
         else
            zz(nx,j+1)=z(imax,j)
         endif
      enddo
c lower (j=1) and upper (j=ny) boundary of ag:
      do i=1,imax
         if (cv(i+1,1).ge.0.) then
            zz(i+1,1)=z2(i)
         else
            zz(i+1,1)=z(i,1)
         endif
         if (cv(i+1,ny-1).le.0.) then 
            zz(i+1,ny)=z4(i)
         else
            zz(i+1,ny)=z(i,jmax)
         endif
      enddo
c call horizontal advection scheme:
c if (pos) positive definite (advpxy) else monotone (advmxy) version.
      if (pos) then
         call advpxy (cu,cv,zz,nx,ny,max(nx,ny))
      else
         call advmxy (cu,cv,zz,nx,ny,max(nx,ny))
      endif
c redistribution of transport quantity from ag to mg.
      do i=1,imax
      do j=1,jmax
         z(i,j)= zz(i+1,j+1)
      enddo
      enddo
      return
      end

      subroutine advmxy (cu,cv,zz,nx,ny,m)
c monotone advection scheme: 1) x-advection, 2) y-advection.
c parameter m is max(nx,ny)
      double precision c(m),z(m),df(m),dfz(m,m)
      dimension cu(nx,ny),cv(nx,ny),zz(nx,ny)
c x-advection
      do 1000 j=2,ny-1
      do 1010 i=1,nx
      z(i)=zz(i,j)
      df(i)=0.
 1010 c(i)=cu(i,j)
      call adv4m (z,c,df,nx)
      do 1020 i=2,nx-1
      zz(i,j)=z(i)
 1020 dfz(i,j)=df(i)
 1000 continue
c y-advection
      do 1030 i=2,nx-1
      do 1040 j=1,ny
      z(j)=zz(i,j)
      df(j)=dfz(i,j)
 1040 c(j)=cv(i,j)
      call adv4m (z,c,df,ny)
      do 1050 j=2,ny-1
 1050 zz(i,j)=z(j)+df(j)
 1030 continue
      return
      end

      subroutine advmyx (cu,cv,zz,nx,ny,m)
c monotone advection scheme: 1) y-advection, 2) x-advection.
c parameter m=max(nx,ny)
c      parameter (m=102)
      double precision c(m),z(m),df(m),dfz(m,m)
      dimension cu(nx,ny),cv(nx,ny),zz(nx,ny)
c y-advection
      do 1000 i=2,nx-1
      do 1010 j=1,ny
      z(j)=zz(i,j)
      df(j)=0.
 1010 c(j)=cv(i,j)
      call adv4m (z,c,df,ny)
      do 1020 j=2,ny-1
      zz(i,j)=z(j)
 1020 dfz(i,j)=df(j)
 1000 continue
c x-advection
      do 1030 j=2,ny-1
      do 1040 i=1,nx
      z(i)=zz(i,j)
      df(i)=dfz(i,j)
 1040 c(i)=cu(i,j)
      call adv4m (z,c,df,nx)
      do 1050 i=2,nx-1
 1050 zz(i,j)=z(i)+df(i)
 1030 continue
      return
      end

      subroutine adv4m (y,c,df,n)
c Area-preserving flux-form advection algorithm.
c Bott, 1989: Monthly Weather Review, 1006-1015, 2633-2636.
c Bott, 1992: Monthly Weather Review, 2592-2602.
c Bott, 1993: Monthly Weather Review, 2637-2641.
c Fourth order monotone version.
c y(i) is transport quantity, input and output.
c Dirichlet boundary conditions are used: y(1)=const, y(n)=const.
c c(i) is Courant number satisfying the CFL criterion, input.
c a0, a1, a2, a3, a4 are coefficients of polynomials in gridbox i.
c At i=1 and i=n first order polynomial,
c at i=2 and i=n-1 second order polynomial,
c at 3<=i<=n-2 fourth order polynomial.
c w(i) are positive definite flux limiters.
c The numerical grid is equidistant.
c c(i), fm(i), fp(i)  are given at the right boundary of grid cell i.
c fm(i), fp(i) are fluxes for c(i)<0 and c(i)>0, respectively.
c fm(i) is flux from gridbox i+1 into gridbox i for c(i)<0,
c fp(i) is flux from gridbox i into gridbox i+1 for c(i)>0.
c fmm(i), fpp(i) are monotone fluxes; df(i) are deformation terms.
      implicit double precision (a-h)
      implicit double precision (o-z)
      dimension a0(n),a1(n),a2(n),a3(n),a4(n),y(n),c(n),
     &          fm(n),fp(n),w(n),fmm(n),fpp(n),df(n)
      a0(2)=(26.*y(2)-y(3)-y(1))/24.
      a1(2)=(y(3)-y(1))/16.
      a2(2)=(y(3)+y(1)-2.*y(2))/48.
      a3(2)=0.
      a4(2)=0.
      do 1000 i=3,n-2
      a0(i)=(9.*(y(i+2)+y(i-2))-116.*(y(i+1)+y(i-1))+2134.*y(i))/1920.
      a1(i)=(-5.*(y(i+2)-y(i-2))+34.*(y(i+1)-y(i-1)))/384.
      a2(i)=(-y(i+2)+12.*(y(i+1)+y(i-1))-22.*y(i)-y(i-2))/384.
      a3(i)=(y(i+2)-2.*(y(i+1)-y(i-1))-y(i-2))/768.
 1000 a4(i)=(y(i+2)-4.*(y(i+1)+y(i-1))+6.*y(i)+y(i-2))/3840.
      a0(n-1)=(26.*y(n-1)-y(n)-y(n-2))/24.
      a1(n-1)=(y(n)-y(n-2))/16.
      a2(n-1)=(y(n)+y(n-2)-2.*y(n-1))/48.
      a3(n-1)=0.
      a4(n-1)=0.
      cl=-dmin1(0.d0,c(n-1))
      fm(n-1)=dmin1(y(n),cl*(y(n)-(1.-cl)*(y(n)-y(n-1))*0.5))
      fmm(n-1)=fm(n-1)
      do 1010 i=n-1,2,-1
      cl=-dmin1(0.d0,c(i))
      x1=1.-2.*cl
      x2=x1*x1
      x3=x1*x2
      ymin=dmin1(y(i),y(i+1))
      ymax=dmax1(y(i),y(i+1))
      fmim=dmax1(0.d0,a0(i)*cl-a1(i)*(1.-x2)+a2(i)*(1.-x3)
     &     -a3(i)*(1.-x1*x3)+a4(i)*(1.-x2*x3))
      fmim=dmin1(fmim,y(i)-ymin+fm(i))
      fmm(i-1)=dmax1(fmim,y(i)-ymax+fm(i))
      fm(i-1)=0.
 1010 if (c(i-1).lt.0.) fm(i-1)=dmax1(0.d0,fmm(i-1)-(cl+c(i-1))*y(i))
      cr=dmax1(0.d0,c(1))
      fp(1)=dmin1(y(1),cr*(y(1)+(1.-cr)*(y(2)-y(1))*0.5))
      fpp(1)=fp(1)
      do 1020 i=2,n-1
      cr=dmax1(0.d0,c(i-1))
      x1=1.-2.*cr
      x2=x1*x1
      x3=x1*x2
      ymin=dmin1(y(i-1),y(i))
      ymax=dmax1(y(i-1),y(i))
      fpi=dmax1(0.d0,a0(i)*cr+a1(i)*(1.-x2)+a2(i)*(1.-x3)
     &    +a3(i)*(1.-x1*x3)+a4(i)*(1.-x2*x3))
      fpi=dmin1(fpi,y(i)-ymin+fp(i-1))
      fpp(i)=dmax1(fpi,y(i)-ymax+fp(i-1))
      fp(i)=0.
 1020 if (c(i).gt.0.) fp(i)=dmax1(0.d0,fpp(i)-(cr-c(i))*y(i))
      do 1030 i=2,n-1
      y(i)=y(i)+df(i)
      x0=dmin1(1.d0,y(i)/(fmm(i-1)+fpp(i)+1.d-15))
      w(i)=dmin1(1.d0,y(i)/(fm(i-1)+fp(i)+1.d-15))
      y(i)=y(i)-(fmm(i-1)+fpp(i))*x0
 1030 df(i)=x0*(fmm(i-1)+fpp(i))-(fm(i-1)+fp(i))*w(i)
      w(1)=w(2)
      w(n)=w(n-1)
      do 1040 i=2,n-1
      y(i)=y(i)+fm(i)*w(i+1)+fp(i-1)*w(i-1)
 1040 continue
      return
      end

      subroutine advpxy (cu,cv,zz,nx,ny,m)
c positive definite advection scheme: 1) x-advection, 2) y-advection.
c parameter m is max(nx,ny)
      double precision c(m),z(m)
      dimension cu(nx,ny),cv(nx,ny),zz(nx,ny)
c x-advection
      do 1000 j=2,ny-1
      do 1010 i=1,nx
      z(i)=zz(i,j)
 1010 c(i)=cu(i,j)
      call adv4p (z,c,nx)
      do 1020 i=2,nx-1
 1020 zz(i,j)=z(i)
 1000 continue
c y-advection
      do 1030 i=2,nx-1
      do 1040 j=1,ny
      z(j)=zz(i,j)
 1040 c(j)=cv(i,j)
      call adv4p (z,c,ny)
      do 1050 j=2,ny-1
 1050 zz(i,j)=z(j)
 1030 continue
      return
      end

      subroutine advpyx (cu,cv,zz,nx,ny,m)
c positive definite advection scheme: 1) y-advection, 2) x-advection.
c parameter m is max(nx,ny)
c      parameter (m=514)
      double precision c(m),z(m)
      dimension cu(nx,ny),cv(nx,ny),zz(nx,ny)
c y-advection
      do 1000 i=2,nx-1
      do 1010 j=1,ny
      z(j)=zz(i,j)
 1010 c(j)=cv(i,j)
      call adv4p (z,c,ny)
      do 1020 j=2,ny-1
 1020 zz(i,j)=z(j)
 1000 continue
c x-advection
      do 1030 j=2,ny-1
      do 1040 i=1,nx
      z(i)=zz(i,j)
 1040 c(i)=cu(i,j)
      call adv4p (z,c,nx)
      do 1050 i=2,nx-1
 1050 zz(i,j)=z(i)
 1030 continue
      return
      end

      subroutine adv4p (y,c,n)
c Area-preserving flux-form advection algorithm.
c Bott, 1989: Monthly Weather Review, 1006-1015, 2633-2636.
c fourth order positive definite version.
c y(i) is transport quantity, input and output.
c Dirichlet boundary conditions are used: y(1)=const, y(n)=const.
c c(i) is Courant number satisfying the CFL criterion, input.
c a0, a1, a2, a3, a4 are coefficients of polynomials in gridbox i.
c At i=1 and i=n first order polynomial,
c at i=2 and i=n-1 second order polynomial,
c at 3<=i<=n-2 fourth order polynomial.
c w(i) are positive definite flux limiters.
c The numerical grid is equidistant.
c c(i), fm(i), fp(i)  are given at the right boundary of grid cell i.
c fm(i), fp(i) are fluxes for c(i)<0 and c(i)>0, respectively.
c fm(i) is flux from gridbox i+1 into gridbox i for c(i)<0,
c fp(i) is flux from gridbox i into gridbox i+1 for c(i)>0.
c      parameter (m=514)
      implicit double precision (a-h)
      implicit double precision (o-z)
      dimension y(n),c(n),fm(n),fp(n),w(n)
      cr=dmax1(0.d0,c(1))
      fp(1)=dmin1(y(1),cr*(y(1)+(1.-cr)*(y(2)-y(1))*0.5))
      w(1)=1.
      a0=(26.*y(2)-y(3)-y(1))/24.
      a1=(y(3)-y(1))/16.
      a2=(y(3)+y(1)-2.*y(2))/48.
      cl=-dmin1(0.d0,c(1))
      x1=1.-2.*cl
      x2=x1*x1
      fm(1)=dmax1(0.d0,a0*cl-a1*(1.-x2)+a2*(1.-x1*x2))
      cr=dmax1(0.d0,c(2))
      x1=1.-2.*cr
      x2=x1*x1
      fp(2)=dmax1(0.d0,a0*cr+a1*(1.-x2)+a2*(1.-x1*x2))
      w(2)=dmin1(1.d0,y(2)/(fm(1)+fp(2)+1.d-15))
      do 1000 i=3,n-2
      a0=(9.*(y(i+2)+y(i-2))-116.*(y(i+1)+y(i-1))+2134.*y(i))/1920.
      a1=(-5.*(y(i+2)-y(i-2))+34.*(y(i+1)-y(i-1)))/384.
      a2=(-y(i+2)+12.*(y(i+1)+y(i-1))-22.*y(i)-y(i-2))/384.
      a3=(y(i+2)-2.*(y(i+1)-y(i-1))-y(i-2))/768.
      a4=(y(i+2)-4.*(y(i+1)+y(i-1))+6.*y(i)+y(i-2))/3840.
      cl=-dmin1(0.d0,c(i-1))
      x1=1.-2.*cl
      x2=x1*x1
      x3=x1*x2
      fm(i-1)=dmax1(0.d0,a0*cl-a1*(1.-x2)+a2*(1.-x3)
     &                   -a3*(1.-x1*x3)+a4*(1.-x2*x3))
      cr=dmax1(0.d0,c(i))
      x1=1.-2.*cr
      x2=x1*x1
      x3=x1*x2
      fp(i)=dmax1(0.d0,a0*cr+a1*(1.-x2)+a2*(1.-x3)
     &                 +a3*(1.-x1*x3)+a4*(1.-x2*x3))
 1000 w(i)=dmin1(1.d0,y(i)/(fm(i-1)+fp(i)+1.d-15))
      a0=(26.*y(n-1)-y(n)-y(n-2))/24.
      a1=(y(n)-y(n-2))/16.
      a2=(y(n)+y(n-2)-2.*y(n-1))/48.
      cl=-dmin1(0.d0,c(n-2))
      x1=1.-2.*cl
      x2=x1*x1
      fm(n-2)=dmax1(0.d0,a0*cl-a1*(1.-x2)+a2*(1.-x1*x2))
      cr=dmax1(0.d0,c(n-1))
      x1=1.-2.*cr
      x2=x1*x1
      fp(n-1)=dmax1(0.d0,a0*cr+a1*(1.-x2)+a2*(1.-x1*x2))
      w(n-1)=dmin1(1.d0,y(n-1)/(fm(n-2)+fp(n-1)+1.d-15))
      cl=-dmin1(0.d0,c(n-1))
      fm(n-1)=dmin1(y(n),cl*(y(n)-(1.-cl)*(y(n)-y(n-1))*0.5))
      w(n)=1.
      do 1010 i=2,n-1
 1010 y(i)=y(i)-(fm(i-1)+fp(i))*w(i)+fm(i)*w(i+1)+fp(i-1)*w(i-1)
      return
      end
