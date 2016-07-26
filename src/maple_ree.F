ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     c
c     Advection schemes                                               c
c     Urs Germann                                                     c
c     Version 5 June 2001                                             c
c     c
c     f2py -c -m maple_ree maple_ree.F
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c-----1----------------------------------------------------------------1

      subroutine ree_epol_slio(r0,re,vx,vy,net,nx,ny,nvx,nvy)
c     previously called ree_epol_sluobl
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     c
c     Semi-lagrangian advection scheme c
c     (upstream, one! bilinear interpolation) c
c     (Code optimized for stationary motion assumption) c
c     Version 29 June 2001                                            c
c     c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c compile code as python module:
c f2py -c -m maple_ree maple_ree.F

Cf2py intent(in) r0
Cf2py intent(out) re
Cf2py intent(in) vx
Cf2py intent(in) vy
Cf2py intent(in) net
Cf2py intent(in) nx
Cf2py intent(in) ny
Cf2py intent(in) nvx
Cf2py intent(in) nvy

      implicit none
      
c     default parameters 
      integer mag
      parameter(mag=10)
      real dx,dy,dte
      parameter(dte=1.,dx=1.,dy=1.)
c     parameter(dte=300.,dx=1000.,dy=1000.)

c     arguments
      integer net,nx,ny,nvx,nvy
      real r0(nx,ny),re(nx,ny,net)
      real vx(nvx,nvy),vy(nvx,nvy)

      real ndx,ndy,ndxh,ndyh
      real a,b,aa,bb,aaa,bbb,rr,ex,ey,exh,eyh
      integer i,j,k,l,ii,jj      
      
c     ---Used in ree_vneix, ree_vneiy
      ndx=float(nx-2*mag)/nvx
      ndy=float(ny-2*mag)/nvy
      ndxh=mag+ndx/2.+0.5
      ndyh=mag+ndy/2.+0.5

      ex=dte/dx
      ey=dte/dy
      exh=dte/dx/2.
      eyh=dte/dy/2.

c     ---Init advected field
      do l=1,net
          do j=1,ny
             do i=1,nx
                re(i,j,l)=0.0
             enddo
          enddo
      enddo

c     ---Start Advection
      do j=1,ny
         do i=1,nx
c     ---Start upstream-semi-lagrange at point i,j
            aaa=float(i)
            bbb=float(j)
c     ---Loop over time steps
            do l=1,net
c     ---Iteration to determine displacement vector
                aa=aaa
                bb=bbb                        

c     ---1st estimate
                call ree_vneix(aa,ii,nvx,ndx,ndxh)                
                call ree_vneiy(bb,jj,nvy,ndy,ndyh)
                call ree_linsp2d(a,aa,bb,
     1                 vx(ii,jj),vx(ii+1,jj),vx(ii,jj+1),
     2                 vx(ii+1,jj+1))
                call ree_linsp2d(b,aa,bb,
     1                 vy(ii,jj),vy(ii+1,jj),vy(ii,jj+1),
     2                 vy(ii+1,jj+1))
                aa=aaa-a*exh
                bb=bbb-b*eyh
                  
c     ---2nd estimate
                call ree_vneix(aa,ii,nvx,ndx,ndxh)
                call ree_vneiy(bb,jj,nvy,ndy,ndyh)
                call ree_linsp2d(a,aa,bb,
     1                 vx(ii,jj),vx(ii+1,jj),vx(ii,jj+1),
     2                 vx(ii+1,jj+1))
                call ree_linsp2d(b,aa,bb,
     1                 vy(ii,jj),vy(ii+1,jj),vy(ii,jj+1),
     2                 vy(ii+1,jj+1))
                aa=aaa-a*exh
                bb=bbb-b*eyh

c     ---Definitive displacement vector
                call ree_vneix(aa,ii,nvx,ndx,ndxh)
                call ree_vneiy(bb,jj,nvy,ndy,ndyh)
                call ree_linsp2d(a,aa,bb,
     1                 vx(ii,jj),vx(ii+1,jj),vx(ii,jj+1),
     2                 vx(ii+1,jj+1))
                call ree_linsp2d(b,aa,bb,
     1                 vy(ii,jj),vy(ii+1,jj),vy(ii,jj+1),
     2                 vy(ii+1,jj+1))
                aaa=aaa-a*ex
                bbb=bbb-b*ey
                aa=aaa
                bb=bbb
c                if(j.eq.256) then
c                 if (i.eq.256) then
c                    if (l.eq.1) then
c                       print*,a,b,aa,bb,ii,jj
c                    endif
c                 endif
c                endif

c     ---Nearest neighbour
                call ree_rneix(aa,ii,nx)
                call ree_rneiy(bb,jj,ny)

c     ---Advect Reflectivity Field  
                if(ii.eq.0.or.jj.eq.0) then 
                    rr=0.0
                else 
                    call ree_linsp2d(rr,aa,bb,r0(ii,jj),
     1                    r0(ii+1,jj),r0(ii,jj+1),
     2                    r0(ii+1,jj+1))
                endif
                re(i,j,l) = rr 
            enddo
         enddo
      enddo
      
      return
      end

c-----1----------------------------------------------------------------1


c     -----Determine lower-left vx,vy neighbour given coordinates aa,bb-
      subroutine ree_vneix(aa,ii,nvx,ndx,ndxh)

      integer ii,nvx
      real ndxh,ndx
      real aa
      
      aa=(aa-ndxh)/ndx + 1.0
      if(aa.lt.1) aa=1.0 
      if(aa.ge.nvx) aa=nvx-1e-5
      ii=aa
      aa=aa-ii
      return
      end
      
      subroutine ree_vneiy(bb,jj,nvy,ndy,ndyh)

      integer jj,nvy
      real ndyh,ndy
      real bb
      bb=(bb-ndyh)/ndy +1.0
      if(bb.lt.1) bb=1.0
      if(bb.ge.nvy) bb=nvy-1e-5
      jj=bb
      bb=bb-jj
      return
      end
      
c     -----Determine lower-left vx,vy neighbour given vx,vy coordinates aa,bb-
      subroutine ree_vneivx(aa,ii,nvx)

      integer ii,nvx
      real aa
      if(aa.lt.1) aa=1.0
      if(aa.ge.nvx) aa=nvx-1e-5
      ii=aa
      aa=aa-ii
      return
      end

      subroutine ree_vneivy(bb,jj,nvy)

      integer jj
      real bb,nvy
      if(bb.lt.1) bb=1.0
      if(bb.ge.nvy) bb=nvy-1e-5
      jj=bb
      bb=bb-jj
      return
      end
      
c     -----Determine lower-left Z neighbour given coordinates aa,bb-----
c     These routines return ii or jj set to zero if coordinates lie 
c     outside the boundaries (cf. advection scheme above)
      subroutine ree_rneix(aa,ii,nx)

      integer ii,nx
      real aa
      if(aa.lt.1) aa=0.0
      if(aa.ge.nx) aa=0.0
      ii=aa
      aa=aa-ii
      return
      end

      subroutine ree_rneiy(bb,jj,ny)

      integer jj,ny
      real bb
      if(bb.lt.1) bb=0.0
      if(bb.ge.ny) bb=0.0
      jj=bb
      bb=bb-jj
      return
      end

c     -----Bilinear spline interpolation-----
      subroutine ree_linsp2d(rr,aa,bb,vll,vlr,vul,vur)
      real rr,aa,bb,vll,vlr,vul,vur
c     The following line is equivalent to 
c     (1-a)(1-b)ll + a(1-b)lr + (ab)ur + (1-a)(b)ul
      rr=vll+aa*(vlr-vll)+bb*(vul-vll)+aa*bb*(vur+vll-vlr-vul)
      return
      end

