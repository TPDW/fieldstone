import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt
from scipy import stats


#------------------------------------------------------------------------------

def density(x,y,y0,rho_alpha):
    lambdaa=1
    k=2*np.pi/lambdaa
    if abs(y-y0)<1e-6:
       val=rho_alpha*np.cos(k*x)#+1.
    else:
       val=0.#+1.
    return val

def sigmayy_th(x,y0):
    lambdaa=1.
    k=2*np.pi/lambdaa
    val=np.cos(k*x)/np.sinh(k)**2*\
       (k*(1.-y0)*np.sinh(k)*np.cosh(k*y0)\
       -k*np.sinh(k*(1.-y0))\
       +np.sinh(k)*np.sinh(k*y0) )
    return val

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3


def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(np.log(x),np.log(y))[0]

def get_regression_x_intercept_difference(h,y1,y2):
    [m1,c1]=stats.linregress(np.log(np.abs(h)),np.log(np.abs(y1)))[0:2]
    [m2,c2]=stats.linregress(np.log(np.abs(h)),np.log(np.abs(y2)))[0:2]
    r1=-c1/m1
    r2=-c2/m2

    return r1-r2


#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 
nelx_list = [16,24,32,40,48,64]
#nelx_list=[64,64]
corner_error_list=[]
L1_norm_list_CBF=[]
L2_norm_list_CBF=[]

L1_norm_list_CN=[]
L2_norm_list_CN=[]

L1_norm_list_LS=[]
L2_norm_list_LS=[]



fig_CN,axes_CN=plt.subplots(2,int(len(nelx_list)/2),figsize=(30,10))

counter_nelx=0

for nelx in nelx_list:
  print("----------------------")
  print("Now Running nelx=" + str(nelx))
  print("----------------------")

  nely=nelx     
  nnx=nelx+1  # number of elements, x direction
  nny=nely+1  # number of elements, y direction
  nnp=nnx*nny  # number of nodes
  nel=nelx*nely  # number of elements, total
  NfemV=nnp*ndofV # number of velocity dofs
  NfemP=nel*ndofP  # number of pressure dofs
  Nfem=NfemV+NfemP # total number of dofs

  gx=0
  gy=-1

  pnormalise=True
  viscosity=1  # dynamic viscosity \mu

  y0=0.5
  rho_alpha=nelx

  hx=Lx/nelx


  eps=1.e-10
  sqrt3=np.sqrt(3.)

  hx = Lx/nelx
  hy = Ly/nely


  #################################################################
  # grid point setup
  #################################################################
  start = time.time()

  x = np.empty(nnp, dtype=np.float64)  # x coordinates
  y = np.empty(nnp, dtype=np.float64)  # y coordinates

  counter = 0
  for j in range(0, nny):
      for i in range(0, nnx):
          x[counter]=i*Lx/float(nelx)
          y[counter]=j*Ly/float(nely)
          counter += 1

  print("setup: grid points: %.3f s" % (time.time() - start))

  #################################################################
  # connectivity
  #################################################################
  start = time.time()

  icon =np.zeros((m, nel),dtype=np.int16)
  counter = 0
  for j in range(0, nely):
      for i in range(0, nelx):
          icon[0, counter] = i + j * (nelx + 1)
          icon[1, counter] = i + 1 + j * (nelx + 1)
          icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
          icon[3, counter] = i + (j + 1) * (nelx + 1)
          counter += 1

  np.savetxt("icon.dat",np.transpose(icon),fmt='%.4i')

  print("setup: connectivity: %.3f s" % (time.time() - start))

  #################################################################
  # define boundary conditions
  #################################################################
  # numbering of faces of the domain
  # +---3---+
  # |       |
  # 0       1
  # |       |
  # +---2---+
  #################################################################
  start = time.time()

  bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
  bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
  on_bd=np.zeros((nnp,4),dtype=np.bool)  # boundary indicator

  for i in range(0, nnp):
      if x[i]<eps:
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         on_bd[i,0]=True
      if x[i]>(Lx-eps):
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         on_bd[i,1]=True
      if y[i]<eps:
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         on_bd[i,2]=True
      if y[i]>(Ly-eps):
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         on_bd[i,3]=True

  print("setup: boundary conditions: %.3f s" % (time.time() - start))

  NfemTr=np.sum(bc_fix)
  print (NfemTr)

  bc_nb=np.zeros(NfemV,dtype=np.int32)  # boundary condition, yes/no

  counter=0
  for i in range(0,NfemV):
      if (bc_fix[i]):
         bc_nb[i]=counter
         counter+=1

  print (np.min(bc_nb))
  print (np.max(bc_nb))

  #################################################################
  # building density array
  #################################################################
  rho = np.empty(nnp, dtype=np.float64)  

  for i in range(0,nnp):
      rho[i]=density(x[i],y[i],y0,rho_alpha)

  #################################################################
  # build FE matrix
  # [ K G ][u]=[f]
  # [GT 0 ][p] [h]
  #################################################################
  start = time.time()

  K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
  G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
  f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
  h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
  b_mat = np.zeros((3,ndofV*m),dtype=np.float64)  # gradient matrix B 
  N     = np.zeros(m,dtype=np.float64)            # shape functions
  dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
  v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
  p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
  c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

  for iel in np.arange(0, nel):

      # set arrays to 0 every loop
      f_el =np.zeros((m*ndofV),dtype=np.float64)
      K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
      G_el=np.zeros((m*ndofV,1),dtype=np.float64)
      h_el=np.zeros((1,1),dtype=np.float64)

      # integrate viscous term at 4 quadrature points
      for iq in [-1, 1]:
          for jq in [-1, 1]:

              # position & weight of quad. point
              rq=iq/sqrt3
              sq=jq/sqrt3
              weightq=1.*1.

              # calculate shape functions
              N[0:m]=NNV(rq,sq)
              dNdr[0:m]=dNNVdr(rq,sq)
              dNds[0:m]=dNNVds(rq,sq)

              # calculate jacobian matrix
              jcb = np.zeros((2, 2),dtype=float)
              for k in range(0,m):
                  jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                  jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                  jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                  jcb[1, 1] += dNds[k]*y[icon[k,iel]]

              # calculate the determinant of the jacobian
              jcob = np.linalg.det(jcb)

              # calculate inverse of the jacobian matrix
              jcbi = np.linalg.inv(jcb)

              # compute dNdx & dNdy
              xq=0.0
              yq=0.0
              rhoq=0.
              for k in range(0, m):
                  xq+=N[k]*x[icon[k,iel]]
                  yq+=N[k]*y[icon[k,iel]]
                  rhoq+=N[k]*rho[icon[k,iel]]
                  dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                  dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

              # construct 3x8 b_mat matrix
              for i in range(0, m):
                  b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                           [0.     ,dNdy[i]],
                                           [dNdy[i],dNdx[i]]]

              # compute elemental a_mat matrix
              K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

              # compute elemental rhs vector
              for i in range(0, m):
                  f_el[ndofV*i  ]+=N[i]*jcob*weightq*rhoq*gx
                  f_el[ndofV*i+1]+=N[i]*jcob*weightq*rhoq*gy
                  G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                  G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

          # end for jq
      # end for iq



      # impose b.c. 
      for k1 in range(0,m):
          for i1 in range(0,ndofV):
              ikk=ndofV*k1          +i1
              m1 =ndofV*icon[k1,iel]+i1
              if bc_fix[m1]:
                 K_ref=K_el[ikk,ikk] 
                 for jkk in range(0,m*ndofV):
                     f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                     K_el[ikk,jkk]=0
                     K_el[jkk,ikk]=0
                 K_el[ikk,ikk]=K_ref
                 f_el[ikk]=K_ref*bc_val[m1]
                 h_el[0]-=G_el[ikk,0]*bc_val[m1]
                 G_el[ikk,0]=0

      # assemble matrix K_mat and right hand side rhs
      for k1 in range(0,m):
          for i1 in range(0,ndofV):
              ikk=ndofV*k1          +i1
              m1 =ndofV*icon[k1,iel]+i1
              for k2 in range(0,m):
                  for i2 in range(0,ndofV):
                      jkk=ndofV*k2          +i2
                      m2 =ndofV*icon[k2,iel]+i2
                      K_mat[m1,m2]+=K_el[ikk,jkk]
              f_rhs[m1]+=f_el[ikk]
              G_mat[m1,iel]+=G_el[ikk,0]
      h_rhs[iel]+=h_el[0]

  print("build FE matrix: %.3f s" % (time.time() - start))

  ######################################################################
  # assemble K, G, GT, f, h into A and rhs
  ######################################################################
  start = time.time()

  if pnormalise:
     a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
     rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
     a_mat[0:NfemV,0:NfemV]=K_mat
     a_mat[0:NfemV,NfemV:Nfem]=G_mat
     a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
     a_mat[Nfem,NfemV:Nfem]=1
     a_mat[NfemV:Nfem,Nfem]=1
  else:
     a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
     rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
     a_mat[0:NfemV,0:NfemV]=K_mat
     a_mat[0:NfemV,NfemV:Nfem]=G_mat
     a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

  rhs[0:NfemV]=f_rhs
  rhs[NfemV:Nfem]=h_rhs

  print("assemble blocks: %.3f s" % (time.time() - start))

  ######################################################################
  # solve system
  ######################################################################
  start = time.time()

  sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs,use_umfpack=True)

  print("solve time: %.3f s" % (time.time() - start))

  ######################################################################
  # put solution into separate x,y velocity arrays
  ######################################################################
  start = time.time()

  u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
  p=sol[NfemV:Nfem]

  print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
  print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

  np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

  print("split vel into u,v: %.3f s" % (time.time() - start))

  ######################################################################
  # compute strainrate 
  ######################################################################
  start = time.time()

  xc = np.zeros(nel,dtype=np.float64)  
  yc = np.zeros(nel,dtype=np.float64)  
  exx = np.zeros(nel,dtype=np.float64)  
  eyy = np.zeros(nel,dtype=np.float64)  
  exy = np.zeros(nel,dtype=np.float64)  

  for iel in range(0,nel):

      rq = 0.0
      sq = 0.0
      N[0:m]=NNV(rq,sq)
      dNdr[0:m]=dNNVdr(rq,sq)
      dNds[0:m]=dNNVds(rq,sq)

      jcb=np.zeros((2,2),dtype=np.float64)
      for k in range(0, m):
          jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
          jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
          jcb[1,0]+=dNds[k]*x[icon[k,iel]]
          jcb[1,1]+=dNds[k]*y[icon[k,iel]]
      jcob=np.linalg.det(jcb)
      jcbi=np.linalg.inv(jcb)

      for k in range(0, m):
          dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
          dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

      for k in range(0, m):
          xc[iel] += N[k]*x[icon[k,iel]]
          yc[iel] += N[k]*y[icon[k,iel]]
          exx[iel] += dNdx[k]*u[icon[k,iel]]
          eyy[iel] += dNdy[k]*v[icon[k,iel]]
          exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+\
                      0.5*dNdx[k]*v[icon[k,iel]]

  print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
  print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
  print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

  np.savetxt('p_elt.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
  np.savetxt('strainrate_elt.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

  print("compute press & sr: %.3f s" % (time.time() - start))

  ######################################################################
  # compute nodal pressure & strainrates : C->N method
  ######################################################################

  q1=np.zeros(nnp,dtype=np.float64)  
  exxn1=np.zeros(nnp,dtype=np.float64)  
  eyyn1=np.zeros(nnp,dtype=np.float64)  
  exyn1=np.zeros(nnp,dtype=np.float64)  
  count=np.zeros(nnp,dtype=np.float64)  

  for iel in range(0,nel):
      q1[icon[0,iel]]+=p[iel]
      q1[icon[1,iel]]+=p[iel]
      q1[icon[2,iel]]+=p[iel]
      q1[icon[3,iel]]+=p[iel]
      exxn1[icon[0,iel]]+=exx[iel]
      exxn1[icon[1,iel]]+=exx[iel]
      exxn1[icon[2,iel]]+=exx[iel]
      exxn1[icon[3,iel]]+=exx[iel]
      eyyn1[icon[0,iel]]+=eyy[iel]
      eyyn1[icon[1,iel]]+=eyy[iel]
      eyyn1[icon[2,iel]]+=eyy[iel]
      eyyn1[icon[3,iel]]+=eyy[iel]
      exyn1[icon[0,iel]]+=exy[iel]
      exyn1[icon[1,iel]]+=exy[iel]
      exyn1[icon[2,iel]]+=exy[iel]
      exyn1[icon[3,iel]]+=exy[iel]
      count[icon[0,iel]]+=1
      count[icon[1,iel]]+=1
      count[icon[2,iel]]+=1
      count[icon[3,iel]]+=1

  q1/=count
  exxn1/=count
  eyyn1/=count
  exyn1/=count

  np.savetxt('q_C-N.ascii',np.array([x,y,q1]).T,header='# x,y,q1')
  np.savetxt('strainrate_C-N.ascii',np.array([x,y,exxn1,eyyn1,exyn1]).T,header='# x,y,exxn1,eyyn1,exyn1')

  #####################################################################
  # compute nodal strain rate - method 3: least squares 
  #####################################################################
  # numbering of elements inside patch
  # -----
  # |3|2|
  # -----
  # |0|1|
  # -----
  # numbering of nodes of the patch
  # 6--7--8
  # |  |  |
  # 3--4--5
  # |  |  |
  # 0--1--2

  q3=np.zeros(nnp,dtype=np.float64)  
  exxn3=np.zeros(nnp,dtype=np.float64)  
  eyyn3=np.zeros(nnp,dtype=np.float64)  
  exyn3=np.zeros(nnp,dtype=np.float64)  

  AA = np.zeros((4,4),dtype=np.float64) 
  BBp  = np.zeros(4,dtype=np.float64) 
  BBxx = np.zeros(4,dtype=np.float64) 
  BByy = np.zeros(4,dtype=np.float64) 
  BBxy = np.zeros(4,dtype=np.float64) 

  counter = 0
  for j in range(0,nely):
      for i in range(0,nelx):
          if i<nelx-1 and j<nely-1:
             iel0=counter
             iel1=counter+1
             iel2=counter+nelx+1
             iel3=counter+nelx

             AA[0,0]=1.       
             AA[1,0]=1.       
             AA[2,0]=1.       
             AA[3,0]=1.        
             AA[0,1]=xc[iel0] 
             AA[1,1]=xc[iel1] 
             AA[2,1]=xc[iel2] 
             AA[3,1]=xc[iel3] 
             AA[0,2]=yc[iel0] 
             AA[1,2]=yc[iel1] 
             AA[2,2]=yc[iel2] 
             AA[3,2]=yc[iel3] 
             AA[0,3]=xc[iel0]*yc[iel0] 
             AA[1,3]=xc[iel1]*yc[iel1] 
             AA[2,3]=xc[iel2]*yc[iel2] 
             AA[3,3]=xc[iel3]*yc[iel3] 

             BBp[0]=p[iel0] 
             BBp[1]=p[iel1] 
             BBp[2]=p[iel2] 
             BBp[3]=p[iel3] 
             solp=sps.linalg.spsolve(sps.csr_matrix(AA),BBp)

             BBxx[0]=exx[iel0] 
             BBxx[1]=exx[iel1] 
             BBxx[2]=exx[iel2] 
             BBxx[3]=exx[iel3] 
             solxx=sps.linalg.spsolve(sps.csr_matrix(AA),BBxx)

             BByy[0]=eyy[iel0] 
             BByy[1]=eyy[iel1] 
             BByy[2]=eyy[iel2] 
             BByy[3]=eyy[iel3] 
             solyy=sps.linalg.spsolve(sps.csr_matrix(AA),BByy)

             BBxy[0]=exy[iel0] 
             BBxy[1]=exy[iel1] 
             BBxy[2]=exy[iel2] 
             BBxy[3]=exy[iel3] 
             solxy=sps.linalg.spsolve(sps.csr_matrix(AA),BBxy)
             
             # node 4 of patch
             ip=icon[2,iel0] 
             q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
             exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
             eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
             exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # node 1 of patch
             ip=icon[1,iel0] 
             if on_bd[ip,2]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # node 3 of patch
             ip=icon[3,iel0] 
             if on_bd[ip,0]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # node 5 of patch
             ip=icon[2,iel1] 
             if on_bd[ip,1]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # node 7 of patch
             ip=icon[3,iel2] 
             if on_bd[ip,3]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # lower left corner of domain
             ip=icon[0,iel0] 
             if on_bd[ip,0] and on_bd[ip,2]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # lower right corner of domain
             ip=icon[1,iel1] 
             if on_bd[ip,1] and on_bd[ip,2]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # upper right corner of domain
             ip=icon[2,iel2] 
             if on_bd[ip,1] and on_bd[ip,3]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

             # lower right corner of domain
             ip=icon[3,iel3] 
             if on_bd[ip,0] and on_bd[ip,3]:
                q3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]
                exxn3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
                eyyn3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
                exyn3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

          counter+=1

          # end if
      # end for i
  # end for j

  print("     -> exxn3 (m,M) %.4e %.4e " %(np.min(exxn3),np.max(exxn3)))
  print("     -> eyyn3 (m,M) %.4e %.4e " %(np.min(eyyn3),np.max(eyyn3)))
  print("     -> exyn3 (m,M) %.4e %.4e " %(np.min(exyn3),np.max(exyn3)))

  np.savetxt('q_LS.ascii',np.array([x,y,q3]).T,header='# x,y,q3')
  np.savetxt('strainrate_ls.ascii',np.array([x,y,exxn3,eyyn3,exyn3]).T,header='# x,y,exxn3,eyyn3,exyn3')

  ######################################################################
  # export elemental sigma_yy
  ######################################################################
  sigmayy_el = np.empty(nel, dtype=np.float64)  
  sigmayy_analytical = np.empty(nnx,dtype=np.float64)  

  for iel in range(1,nel):
      sigmayy_el[iel]=(-p[iel]+2.*viscosity*eyy[iel])

  np.savetxt('sigmayy_el.ascii',np.array([xc[nel-nelx:nel],\
                                          (sigmayy_el[nel-nelx:nel])/hx,\
                                         ]).T,header='# xc,sigmayy')

  np.savetxt('sigmayy_C-N.ascii',np.array([x[nnp-nnx:nnp],\
                                          (-q1[nnp-nnx:nnp]+2.*viscosity*eyyn1[nnp-nnx:nnp])/hx,\
                                          (-q1[nnp-nnx:nnp]+2.*viscosity*eyyn1[nnp-nnx:nnp])/hx - \
                                           sigmayy_th(x[nnp-nnx:nnp],y0),\
                                          ]).T,header='# x,sigmayy,error')

  np.savetxt('sigmayy_LS.ascii',np.array([x[nnp-nnx:nnp],\
                                          (-q3[nnp-nnx:nnp]+2.*viscosity*eyyn3[nnp-nnx:nnp])/hx,\
                                          (-q3[nnp-nnx:nnp]+2.*viscosity*eyyn3[nnp-nnx:nnp])/hx - \
                                           sigmayy_th(x[nnp-nnx:nnp],y0),\
                                          ]).T,header='# x,sigmayy,error')



  #counter=0
  #for i in range(nnp-nnx,nnp):
  #    sigmayy_analytical[counter]=sigmayy_th(x[i],y0)
  #    counter+=1
  #np.savetxt('sigmayy_analytical.ascii',np.array([x[nnp-nnx:nnp],sigmayy_analytical]).T,header='# x,sigmayy')

  print("     -> sigmayy_el       (N-E) %6f " % ((sigmayy_el[nel-1])/hx) ) 
  print("     -> sigmayy_nod C->N (N-E) %6f " % ((-q1[nnp-1]+2.*viscosity*eyyn1[nnp-1])/hx) ) 
  print("     -> sigmayy_nod LS   (N-E) %6f " % ((-q3[nnp-1]+2.*viscosity*eyyn3[nnp-1])/hx) )

  #####################################################################
  # Consistent Boundary Flux method
  #####################################################################
  # we wish to compute sigma_yy on the top boundary only
  # this boundary counts nnx nodes, each with 1 dof for ty

  M_prime = np.zeros((NfemTr,NfemTr),np.float64)
  rhs_cbf = np.zeros(NfemTr,np.float64)
  tx = np.zeros(nnp,np.float64)
  ty = np.zeros(nnp,np.float64)

  M_prime_el =(hx/2.)*np.array([ \
  [2./3.,1./3.],\
  [1./3.,2./3.]])

  for iel in range(0,nel):

      #-----------------------
      # compute Kel, Gel, f
      #-----------------------

      f_el =np.zeros((m*ndofV),dtype=np.float64)
      K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
      G_el=np.zeros((m*ndofV,1),dtype=np.float64)
      rhs_el =np.zeros((m*ndofV),dtype=np.float64)

      # integrate viscous term at 4 quadrature points
      for iq in [-1, 1]:
          for jq in [-1, 1]:

              # position & weight of quad. point
              rq=iq/sqrt3
              sq=jq/sqrt3
              weightq=1.*1.

              # calculate shape functions
              N[0:m]=NNV(rq,sq)
              dNdr[0:m]=dNNVdr(rq,sq)
              dNds[0:m]=dNNVds(rq,sq)

              # calculate jacobian matrix
              jcb = np.zeros((2, 2),dtype=float)
              for k in range(0,m):
                  jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                  jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                  jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                  jcb[1, 1] += dNds[k]*y[icon[k,iel]]
              jcob = np.linalg.det(jcb)
              jcbi = np.linalg.inv(jcb)

              # compute dNdx & dNdy
              rhoq=0.
              for k in range(0, m):
                  rhoq+=N[k]*rho[icon[k,iel]]
                  dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                  dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

              # construct 3x8 b_mat matrix
              for i in range(0, m):
                  b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                           [0.     ,dNdy[i]],
                                           [dNdy[i],dNdx[i]]]

              # compute elemental a_mat matrix
              K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob
              #print(rhoq)
              # compute elemental rhs vector
              for i in range(0, m):
                  f_el[ndofV*i  ]+=N[i]*jcob*weightq*rhoq*gx
                  f_el[ndofV*i+1]+=N[i]*jcob*weightq*rhoq*gy
                  G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                  G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

          # end for jq
      # end for iq

      #-----------------------
      # compute total rhs
      #-----------------------

      v_el = np.array([u[icon[0,iel]],v[icon[0,iel]],\
                       u[icon[1,iel]],v[icon[1,iel]],\
                       u[icon[2,iel]],v[icon[2,iel]],\
                       u[icon[3,iel]],v[icon[3,iel]] ])

      #print(f_el)


      rhs_el=-f_el+K_el.dot(v_el)+G_el[:,0]*p[iel]
      #-----------------------
      # assemble 
      #-----------------------

      #boundary 0-1 : x,y dofs
      for i in range(0,ndofV):
          idof0=2*icon[0,iel]+i
          idof1=2*icon[1,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[0+i]   
             rhs_cbf[idofTr1]+=rhs_el[2+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

      #boundary 1-2 : x,y dofs
      for i in range(0,ndofV):
          idof0=2*icon[1,iel]+i
          idof1=2*icon[2,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[2+i]   
             rhs_cbf[idofTr1]+=rhs_el[4+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

      #boundary 2-3 : x,y dofs
      for i in range(0,ndofV):
          idof0=2*icon[2,iel]+i
          idof1=2*icon[3,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[4+i]   
             rhs_cbf[idofTr1]+=rhs_el[6+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

      #boundary 3-0 : x,y dofs
      for i in range(0,ndofV):
          idof0=2*icon[3,iel]+i
          idof1=2*icon[0,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[6+i]   
             rhs_cbf[idofTr1]+=rhs_el[0+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

  # end for iel

  #np.savetxt("rhs_cbf.ascii",rhs_cbf)
  #matfile=open("matrix.ascii","w")
  #for i in range(0,NfemTr):
  #    for j in range(0,NfemTr):
  #        if abs(M_prime[i,j])>1e-16:
  #           matfile.write(" %d %d %e \n " % (i,j,M_prime[i,j]))
  #matfile.close()

  print("     -> M_prime (m,M) %.4e %.4e " %(np.min(M_prime),np.max(M_prime)))
  print("     -> rhs_cbf (m,M) %.4e %.4e " %(np.min(rhs_cbf),np.max(rhs_cbf)))

  sol=sps.linalg.spsolve(sps.csr_matrix(M_prime),rhs_cbf)#,use_umfpack=True)

  for i in range(0,nnp):
      idof=2*i+0
      if bc_fix[idof]:
         tx[i]=sol[bc_nb[idof]]
      idof=2*i+1
      if bc_fix[idof]:
         ty[i]=sol[bc_nb[idof]]

  np.savetxt('sigmayy_cbf.ascii',np.array([x[nnp-nnx:nnp],ty[nnp-nnx:nnp]]).T,header='# x,sigmayy')

  print("     -> tx (m,M) %.4e %.4e " %(np.min(tx),np.max(tx)))
  print("     -> ty (m,M) %.4e %.4e " %(np.min(ty),np.max(ty)))

  np.savetxt('tractions.ascii',np.array([x,y,tx,ty]).T,header='# x,y,tx,ty')


  #Print the y-traction on the top left corner, along with the theoretical value and the error
  print("CBF Traction, Theoretical Traction, Percentage Error")
  print(ty[nnp-nnx],sigmayy_th(x[nnp-nnx],y0),100*abs(ty[nnp-nnx]-sigmayy_th(x[nnp-nnx],y0))/sigmayy_th(x[nnp-nnx],y0))


  ##### Plot the tractions on the upper surface###############


  axes_CN[counter_nelx%2][int(counter_nelx/2)].plot(sigmayy_th(x[nnp-nnx:nnp],y0),label="Theoretical")
  axes_CN[counter_nelx%2][int(counter_nelx/2)].plot(ty[nnp-nnx:nnp],label="CBF")
  axes_CN[counter_nelx%2][int(counter_nelx/2)].plot(-q1[nnp-nnx:nnp]+2.*viscosity*eyyn1[nnp-nnx:nnp],label="CN")
  axes_CN[counter_nelx%2][int(counter_nelx/2)].set_xlabel("nelx="+str(nelx))
  axes_CN[counter_nelx%2][int(counter_nelx/2)].legend()




  counter_nelx+=1



  ############################################################################
  ####### Calculate the L1 and L2 norms for ty on the upper surface ##########
  ############################################################################

  L1_norm_CBF=0
  L2_norm_CBF=0
  L1_norm_CN=0
  L2_norm_CN=0
  L1_norm_LS=0
  L2_norm_LS=0

  for iel in range(nel-nelx,nel):
      for iq in [-1,1]:
          rq=iq/sqrt3

          N0=0.5*(1-rq)
          N1=0.5*(1+rq)
          wq=1*1

          xq=N0*x[icon[3,iel]]+N1*x[icon[2,iel]]
          tyq_CBF=N0*ty[icon[3,iel]]+N1*ty[icon[2,iel]]
          tyq_CN=N0*(2*viscosity*eyyn1[icon[3,iel]]-q1[icon[3,iel]])+N1*(2*viscosity*eyyn1[icon[2,iel]]-q1[icon[2,iel]])
          tyq_LS=N0*(2*viscosity*eyyn3[icon[3,iel]]-q3[icon[3,iel]])+N1*(2*viscosity*eyyn3[icon[2,iel]]-q3[icon[2,iel]])



          L1_norm_CBF+=abs(tyq_CBF-sigmayy_th(xq,y0))*hx*wq
          L2_norm_CBF+=(tyq_CBF-sigmayy_th(xq,y0))**2*hx*wq

          L1_norm_CN+=abs(tyq_CN-sigmayy_th(xq,y0))*hx*wq
          L2_norm_CN+=(tyq_CN-sigmayy_th(xq,y0))**2*hx*wq

          L1_norm_LS+=abs(tyq_LS-sigmayy_th(xq,y0))*hx*wq
          L2_norm_LS+=(tyq_LS-sigmayy_th(xq,y0))**2*hx*wq



  L1_norm_list_CBF.append(L1_norm_CBF)
  L2_norm_list_CBF.append(np.sqrt(L2_norm_CBF))
  L1_norm_list_CN.append(L1_norm_CN)
  L2_norm_list_CN.append(np.sqrt(L2_norm_CN))
  L1_norm_list_LS.append(L1_norm_LS)
  L2_norm_list_LS.append(np.sqrt(L2_norm_LS))

fig_CN.savefig("nelx_plots.pdf")


hx_list=1/(np.array(nelx_list)+1)
log_hx=np.log(np.abs(hx_list))

fig,ax=plt.subplots()

#ax.plot(log_hx,np.log(np.abs(corner_error_list)),label="corner CBF")
ax.plot(log_hx,np.log(L1_norm_list_CBF),label="L1 CBF")
ax.plot(log_hx,np.log(L2_norm_list_CBF),label="L2 CBF")
ax.plot(log_hx,np.log(L1_norm_list_CN),label="L1 CN")
ax.plot(log_hx,np.log(L2_norm_list_CN),label="L2 CN")
ax.plot(log_hx,np.log(L1_norm_list_LS),label="L1 LS")
ax.plot(log_hx,np.log(L2_norm_list_LS),label="L2 LS")
ax.grid()
ax.legend()
ax.set_xlabel("log(1/nnx)")
ax.set_ylabel("Norm Magnitude")
ax.set_title("Convergence Rates for Traction Norms")
fig.savefig("convergence.pdf")


######Now do the regressions

L1_norm_CBF_regression = get_regression(hx_list,L1_norm_list_CBF)
L1_norm_CN_regression  = get_regression(hx_list,L1_norm_list_CN)
L1_norm_LS_regression  = get_regression(hx_list,L1_norm_list_LS)

L2_norm_CBF_regression = get_regression(hx_list,L2_norm_list_CBF)
L2_norm_CN_regression  = get_regression(hx_list,L2_norm_list_CN)
L2_norm_LS_regression  = get_regression(hx_list,L2_norm_list_LS)


print("L1 norm convergence rates")
print("CBF:" + str(L1_norm_CBF_regression))
print("CN :" + str(L1_norm_CN_regression))
print("LS :" + str(L1_norm_LS_regression))

print("L2 norm convergence rates")
print("CBF:" + str(L2_norm_CBF_regression))
print("CN :" + str(L2_norm_CN_regression))
print("LS :" + str(L2_norm_LS_regression))

L1_CBF_CN_difference = get_regression_x_intercept_difference(hx_list,L1_norm_list_CBF,L1_norm_list_CN)
L1_CBF_LS_difference = get_regression_x_intercept_difference(hx_list,L1_norm_list_CBF,L1_norm_list_LS)
L1_LS_CN_difference  = get_regression_x_intercept_difference(hx_list,L1_norm_list_LS,L1_norm_list_CN)

L2_CBF_CN_difference = get_regression_x_intercept_difference(hx_list,L2_norm_list_CBF,L2_norm_list_CN)
L2_CBF_LS_difference = get_regression_x_intercept_difference(hx_list,L2_norm_list_CBF,L2_norm_list_LS)
L2_LS_CN_difference  = get_regression_x_intercept_difference(hx_list,L2_norm_list_LS,L2_norm_list_CN)

print("L1 LS CN ",L1_LS_CN_difference)
print("L1 CN CBF ",L1_CBF_CN_difference)
print("L1 CBF LS ",L1_CBF_LS_difference)

print("L2 LS CN ",L2_LS_CN_difference)
print("L2 CN CBF ",L2_CBF_CN_difference)
print("L2 CBF LS ",L2_CBF_LS_difference)




# #####################################################################
# # plot of solution
# #####################################################################

# filename = 'solution.vtu'
# vtufile=open(filename,"w")
# vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
# vtufile.write("<UnstructuredGrid> \n")
# vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
# #####
# vtufile.write("<Points> \n")
# vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
# vtufile.write("</DataArray>\n")
# vtufile.write("</Points> \n")
# #####
# vtufile.write("<CellData Scalars='scalars'>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % exx[iel])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % eyy[iel])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % exy[iel])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % p[iel])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % sigmayy_el[iel])
# vtufile.write("</DataArray>\n")

# vtufile.write("</CellData>\n")
# #####
# vtufile.write("<PointData Scalars='scalars'>\n")
# #--
# vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='tractions' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e %20e %20e \n" %(tx[i],ty[i],0.))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='q (C-N)' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e \n" %q1[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='q (LS)' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e \n" %q3[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='eyy (C-N)' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e \n" %eyyn1[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='eyy (LS)' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e \n" %eyyn3[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e \n" %rho[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("</PointData>\n")
# #####
# vtufile.write("<Cells>\n")
# #--
# vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%d \n" %((iel+1)*m))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
# for iel in range (0,nel):
#     vtufile.write("%d \n" %9)
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("</Cells>\n")
# #####
# vtufile.write("</Piece>\n")
# vtufile.write("</UnstructuredGrid>\n")
# vtufile.write("</VTKFile>\n")
# vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


