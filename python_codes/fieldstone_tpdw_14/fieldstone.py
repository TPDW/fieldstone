import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt
from scipy import stats


#TODO: Implement theoretical tractions, plot along the correct axes/boundaries to compare CBF accuracy

#------------------------------------------------------------------------------

def bx(x, y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val
def by(x, y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
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



def sigma_xx(x,y):
  return 2*x**2*(2*x - 2)*(4*y**3 - 6*y**2 + 2*y) + 4*x*(-x + 1)**2*(4*y**3 - 6*y**2 + 2*y) - x*(-x + 1) + 1/6

def sigma_xy(x,y):
  return x**2*(-x + 1)**2*(12*y**2 - 12*y + 2) - y**2*(-y + 1)**2*(12*x**2 - 12*x + 2)

def sigma_yy(x,y):
  return -x*(-x + 1) - 2*y**2*(2*y - 2)*(4*x**3 - 6*x**2 + 2*x) - 4*y*(-y + 1)**2*(4*x**3 - 6*x**2 + 2*x) + 1/6

def sigma_yx(x,y):
  return sigma_xy(x,y)

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)

    if (limitX != 0.0):
       axes[plotX][plotY].set_xlim(0,limitX)

    if (limitY != 0.0):
       axes[plotX][plotY].set_ylim(0,limitY)

    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

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

nelx_list = [16,24,32,40,48,56,64]
#nelx_list=[64,64]

corner_error_list=[[] for Null in range(8)]
L1_norm_list_CBF=[[] for Null in range(8)]
L2_norm_list_CBF=[[] for Null in range(8)]

L1_norm_list_CN=[[] for Null in range(8)]
L2_norm_list_CN=[[] for Null in range(8)]

L1_norm_list_LS=[[] for Null in range(8)]
L2_norm_list_LS=[[] for Null in range(8)]

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


  pnormalise=False
  sparse=False
  viscosity=1  # dynamic viscosity \mu

  penalty=1.e7  # penalty coefficient value


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
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         on_bd[i,0]=True
      if x[i]>(Lx-eps):
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         on_bd[i,1]=True
      if y[i]<eps:
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         on_bd[i,2]=True
      if y[i]>(Ly-eps):
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
         on_bd[i,3]=True


  start = time.time()

  bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
  bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
  for i in range(0, nnp):
      if x[i]<eps:
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
      if x[i]>(Lx-eps):
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
      if y[i]<eps:
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
      if y[i]>(Ly-eps):
         bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
         bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

  print("setup: boundary conditions: %.3f s" % (time.time() - start))

  NfemTr=np.sum(bc_fix)

  bc_nb=np.zeros(NfemV,dtype=np.int32)  # boundary condition number

  counter=0
  for i in range(0,NfemV):
      if (bc_fix[i]):
         bc_nb[i]=counter
         counter+=1


  #################################################################
  # build FE matrix
  # [ K G ][u]=[f]
  # [GT 0 ][p] [h]
  #################################################################
  start = time.time()

  a_mat = np.zeros((NfemV,NfemV),dtype=np.float64)  # matrix of Ax=b
  b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
  rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side of Ax=b
  N     = np.zeros(m,dtype=np.float64)            # shape functions
  dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
  u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
  v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
  k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
  c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 


  for iel in range(0, nel):

      # set 2 arrays to 0 every loop
      b_el = np.zeros(m * ndofV)
      a_el = np.zeros((m * ndofV, m * ndofV), dtype=float)

      # integrate viscous term at 4 quadrature points
      for iq in [-1, 1]:
          for jq in [-1, 1]:

              # position & weight of quad. point
              rq=iq/sqrt3
              sq=jq/sqrt3
              wq=1.*1.

              # calculate shape functions
              N[0]=0.25*(1.-rq)*(1.-sq)
              N[1]=0.25*(1.+rq)*(1.-sq)
              N[2]=0.25*(1.+rq)*(1.+sq)
              N[3]=0.25*(1.-rq)*(1.+sq)

              # calculate shape function derivatives
              dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
              dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
              dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
              dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

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
              for k in range(0, m):
                  xq+=N[k]*x[icon[k,iel]]
                  yq+=N[k]*y[icon[k,iel]]
                  dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                  dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

              # construct 3x8 b_mat matrix
              for i in range(0, m):
                  b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                           [0.     ,dNdy[i]],
                                           [dNdy[i],dNdx[i]]]

              # compute elemental a_mat matrix
              a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

              # compute elemental rhs vector
              for i in range(0, m):
                  b_el[2*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                  b_el[2*i+1]+=N[i]*jcob*wq*by(xq,yq)

      # integrate penalty term at 1 point
      rq=0.
      sq=0.
      wq=2.*2.

      N[0]=0.25*(1.-rq)*(1.-sq)
      N[1]=0.25*(1.+rq)*(1.-sq)
      N[2]=0.25*(1.+rq)*(1.+sq)
      N[3]=0.25*(1.-rq)*(1.+sq)

      dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
      dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
      dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
      dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

      # compute the jacobian
      jcb=np.zeros((2,2),dtype=float)
      for k in range(0, m):
          jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
          jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
          jcb[1,0]+=dNds[k]*x[icon[k,iel]]
          jcb[1,1]+=dNds[k]*y[icon[k,iel]]

      # calculate determinant of the jacobian
      jcob = np.linalg.det(jcb)

      # calculate the inverse of the jacobian
      jcbi = np.linalg.inv(jcb)

      # compute dNdx and dNdy
      for k in range(0,m):
          dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
          dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

      # compute gradient matrix
      for i in range(0,m):
          b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                [0.     ,dNdy[i]],
                                [dNdy[i],dNdx[i]]]

      # compute elemental matrix
      a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

      # assemble matrix a_mat and right hand side rhs
      for k1 in range(0,m):
          for i1 in range(0,ndofV):
              ikk=ndofV*k1          +i1
              m1 =ndofV*icon[k1,iel]+i1
              for k2 in range(0,m):
                  for i2 in range(0,ndofV):
                      jkk=ndofV*k2          +i2
                      m2 =ndofV*icon[k2,iel]+i2
                      a_mat[m1,m2]+=a_el[ikk,jkk]
              rhs[m1]+=b_el[ikk]

  print("build FE matrix: %.3f s" % (time.time() - start))



  #################################################################
  # impose boundary conditions
  #################################################################
  start = time.time()

  for i in range(0, NfemV):
      if bc_fix[i]:
         a_matref = a_mat[i,i]
         for j in range(0,NfemV):
             rhs[j]-= a_mat[i, j] * bc_val[i]
             a_mat[i,j]=0.
             a_mat[j,i]=0.
             a_mat[i,i] = a_matref
         rhs[i]=a_matref*bc_val[i]

  #print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat),np.max(a_mat)))
  #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

  print("impose b.c.: %.3f s" % (time.time() - start))


  ######################################################################
  # solve system
  ######################################################################
  start = time.time()

  if sparse:
    sparse_matrix = A_sparse.tocsr()
  else:
    sparse_matrix = sps.csr_matrix(a_mat)

  print("sparse matrix time: %.3f s" % (time.time()-start))

  start=time.time()

  sol=sps.linalg.spsolve(sparse_matrix,rhs,use_umfpack=True)

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

  #####################################################################
  # retrieve pressure
  #####################################################################
  start = time.time()

  xc = np.zeros(nel,dtype=np.float64)  
  yc = np.zeros(nel,dtype=np.float64)  
  p  = np.zeros(nel,dtype=np.float64)  
  exx = np.zeros(nel,dtype=np.float64)  
  eyy = np.zeros(nel,dtype=np.float64)  
  exy = np.zeros(nel,dtype=np.float64)  

  for iel in range(0,nel):

      rq = 0.0
      sq = 0.0
      wq = 2.0 * 2.0

      N[0]=0.25*(1.-rq)*(1.-sq)
      N[1]=0.25*(1.+rq)*(1.-sq)
      N[2]=0.25*(1.+rq)*(1.+sq)
      N[3]=0.25*(1.-rq)*(1.+sq)

      dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
      dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
      dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
      dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

      jcb=np.zeros((2,2),dtype=float)
      for k in range(0, m):
          jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
          jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
          jcb[1,0]+=dNds[k]*x[icon[k,iel]]
          jcb[1,1]+=dNds[k]*y[icon[k,iel]]

      # calculate determinant of the jacobian
      jcob=np.linalg.det(jcb)

      # calculate the inverse of the jacobian
      jcbi=np.linalg.inv(jcb)

      for k in range(0, m):
          dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
          dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

      for k in range(0, m):
          xc[iel] += N[k]*x[icon[k,iel]]
          yc[iel] += N[k]*y[icon[k,iel]]
          exx[iel] += dNdx[k]*u[icon[k,iel]]
          eyy[iel] += dNdy[k]*v[icon[k,iel]]
          exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

      p[iel]=-penalty*(exx[iel]+eyy[iel])

  print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
  print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
  print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
  print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

  np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')

  np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

  print("compute press & sr: %.3f s" % (time.time() - start))




  ######################################################################
  # compute nodal pressure
  ######################################################################

  q=np.zeros(nnp,dtype=np.float64)  
  count=np.zeros(nnp,dtype=np.float64)  

  for iel in range(0,nel):
      q[icon[0,iel]]+=p[iel]
      q[icon[1,iel]]+=p[iel]
      q[icon[2,iel]]+=p[iel]
      q[icon[3,iel]]+=p[iel]
      count[icon[0,iel]]+=1
      count[icon[1,iel]]+=1
      count[icon[2,iel]]+=1
      count[icon[3,iel]]+=1

  q=q/count


  p_smoothed = np.zeros(nel,dtype=np.float64)

  for iel in range(0,nel):
    p_smoothed[iel] = np.sum(q[icon[:,iel]])/m




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

  sxxn1=2*exxn1-q1
  syyn1=2*eyyn1-q1
  sxyn1=exyn1

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

  # ######################################################################
  # # export elemental sigma_yy
  # ######################################################################
  sigmayy_el = np.empty(nel, dtype=np.float64)  
  sigmayy_analytical = np.empty(nnx,dtype=np.float64)  

  for iel in range(1,nel):
      sigmayy_el[iel]=(-p[iel]+2.*viscosity*eyy[iel])

  np.savetxt('sigmayy_el.ascii',np.array([xc[nel-nelx:nel],\
                                          (sigmayy_el[nel-nelx:nel])/hx,\
                                         ]).T,header='# xc,sigmayy')

  np.savetxt('sigmayy_C-N.ascii',np.array([x[nnp-nnx:nnp],\
                                          (-q1[nnp-nnx:nnp]+2.*viscosity*eyyn1[nnp-nnx:nnp])/hx,\
                                          ]).T,header='# x,sigmayy')

  np.savetxt('sigmayy_LS.ascii',np.array([x[nnp-nnx:nnp],\
                                          (-q3[nnp-nnx:nnp]+2.*viscosity*eyyn3[nnp-nnx:nnp])/hx,\
                                          ]).T,header='# x,sigmayy')





  print("     -> sigmayy_el       (N-E) %6f " % ((sigmayy_el[nel-1])/hx) ) 
  print("     -> sigmayy_nod C->N (N-E) %6f " % ((-q1[nnp-1]+2.*viscosity*eyyn1[nnp-1])/hx) ) 
  print("     -> sigmayy_nod LS   (N-E) %6f " % ((-q3[nnp-1]+2.*viscosity*eyyn3[nnp-1])/hx) )

  #####################################################################
  # Consistent Boundary Flux method
  #####################################################################


  M_prime = np.zeros((NfemTr,NfemTr),np.float64)
  rhs_cbf = np.zeros(NfemTr,np.float64)
  tx = np.zeros(nnp,np.float64)
  ty = np.zeros(nnp,np.float64)

  use_ML=True
  if use_ML:
    M_prime_el =(hx/2.)*np.array([ \
    [1,0],\
    [0,1]])
  else:
    M_prime_el =(hx/2.)*np.array([ \
    [2./3.,1./3.],\
    [1./3.,2./3.]])

  CBF_use_smoothed_pressure=False

  for iel in range(0,nel):
      #print(iel)
       # set 2 arrays to 0 every loop
      b_el = np.zeros(m * ndofV)
      a_el = np.zeros((m * ndofV, m * ndofV), dtype=float)

      # integrate viscous term at 4 quadrature points
      for iq in [-1, 1]:
          for jq in [-1, 1]:

              # position & weight of quad. point
              rq=iq/sqrt3
              sq=jq/sqrt3
              wq=1.*1.

              # calculate shape functions
              N[0]=0.25*(1.-rq)*(1.-sq)
              N[1]=0.25*(1.+rq)*(1.-sq)
              N[2]=0.25*(1.+rq)*(1.+sq)
              N[3]=0.25*(1.-rq)*(1.+sq)

              # calculate shape function derivatives
              dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
              dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
              dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
              dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

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
              for k in range(0, m):
                  xq+=N[k]*x[icon[k,iel]]
                  yq+=N[k]*y[icon[k,iel]]
                  dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                  dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

              # construct 3x8 b_mat matrix
              for i in range(0, m):
                  b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                           [0.     ,dNdy[i]],
                                           [dNdy[i],dNdx[i]]]

              # compute elemental a_mat matrix
              a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

              # compute elemental rhs vector
              for i in range(0, m):
                  b_el[2*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                  b_el[2*i+1]+=N[i]*jcob*wq*by(xq,yq)

      # integrate penalty term at 1 point
      rq=0.
      sq=0.
      wq=2.*2.

      N[0]=0.25*(1.-rq)*(1.-sq)
      N[1]=0.25*(1.+rq)*(1.-sq)
      N[2]=0.25*(1.+rq)*(1.+sq)
      N[3]=0.25*(1.-rq)*(1.+sq)

      dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
      dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
      dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
      dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

      # compute the jacobian
      jcb=np.zeros((2,2),dtype=float)
      for k in range(0, m):
          jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
          jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
          jcb[1,0]+=dNds[k]*x[icon[k,iel]]
          jcb[1,1]+=dNds[k]*y[icon[k,iel]]

      # calculate determinant of the jacobian
      jcob = np.linalg.det(jcb)

      # calculate the inverse of the jacobian
      jcbi = np.linalg.inv(jcb)

      # compute dNdx and dNdy
      for k in range(0,m):
          dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
          dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

      # compute gradient matrix
      for i in range(0,m):
          b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                [0.     ,dNdy[i]],
                                [dNdy[i],dNdx[i]]]

      # compute elemental matrix
      a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob
      #-----------------------
      # compute total rhs
      #-----------------------

      v_el = np.array([u[icon[0,iel]],v[icon[0,iel]],\
                       u[icon[1,iel]],v[icon[1,iel]],\
                       u[icon[2,iel]],v[icon[2,iel]],\
                       u[icon[3,iel]],v[icon[3,iel]] ])


      rhs_el = a_el.dot(v_el) -b_el

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

  np.savetxt("tractions_x.ascii",tx)
  np.savetxt("tractions_y.ascii",ty)

  ######################################################################
  ##### Output plots of tractions
  ######################################################################

  fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

  #ax1 contains the lower tractions I guess

  ax1.plot(tx[:nnx],label="$t_x$ (CBF)")
  ax1.plot(ty[:nnx],label="$t_y$ (CBF)")
  ax1.plot(-sigma_xy(x[:nnx],0),label="$t_x$ analytical")
  ax1.plot(-sigma_yy(x[:nnx],0),label="$t_y$ analytical")
  ax1.legend()
  ax1.set_title("Lower Boundary")

  ax2.plot(tx[(nnp-nnx):],label="$t_x$ (CBF)")
  ax2.plot(ty[(nnp-nnx):],label="$t_y$ (CBF)")
  ax2.plot(sigma_xy(x[(nnp-nnx):],1),label="$t_x$ analytical")
  ax2.plot(sigma_yy(x[(nnp-nnx):],1),label="$t_y$ analytical")
  ax2.legend()
  ax2.set_title("Upper Boundary")


  #there's probably a better way of doing this
  #but I'm not at my best so here's a bit of a hack

  left_ty = []
  right_ty= []
  left_tx = []
  right_tx= []
  right_y = []
  left_y  = []

  for i in range(0,nnp):
    if x[i]<eps:
      left_tx.append(tx[i])
      left_ty.append(ty[i])
      left_y.append(y[i])
    if (Lx-x[i])<eps:
      right_tx.append(tx[i])
      right_ty.append(ty[i])
      right_y.append(y[i])

  right_y=np.array(right_y)
  left_y=np.array(left_y)
  left_tx=np.array(left_tx)
  right_tx=np.array(right_tx)
  left_ty=np.array(left_ty)
  right_ty=np.array(right_ty)

  ax3.plot(right_tx,label="$t_x$ (CBF)")
  ax3.plot(right_ty,label="$t_y$ (CBF)")
  ax3.plot(sigma_xx(1,right_y),label="$t_x$ analytical")
  ax3.plot(sigma_yx(1,right_y),label="$t_y$ analytical")
  ax3.legend()
  ax3.set_title("Right Boundary")

  ax4.plot(left_tx,label="$t_x$ (CBF)")
  ax4.plot(left_ty,label="$t_y$ (CBF)")
  ax4.plot(-sigma_xx(0,left_y),label="$t_x$ analytical")
  ax4.plot(-sigma_yx(0,left_y),label="$t_y$ analytical")
  ax4.legend()
  ax4.set_title("Left Boundary")

  fig.savefig("tractions_CBF"+str(nelx)+".pdf")

  np.savetxt('sigmayy_cbf.ascii',np.array([x[nnp-nnx:nnp],ty[nnp-nnx:nnp]]).T,header='# x,sigmayy')

  print("     -> tx (m,M) %.4e %.4e " %(np.min(tx),np.max(tx)))
  print("     -> ty (m,M) %.4e %.4e " %(np.min(ty),np.max(ty)))

  np.savetxt('tractions.ascii',np.array([x,y,tx,ty]).T,header='# x,y,tx,ty')
  ##################################################
  # Plot the C->N tractions and CBF
  ##################################################

  fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

  #ax1 contains the lower tractions I guess

  ax1.plot(-2*exyn1[:nnx],label="$t_x$ (C->N)")
  ax1.plot(-2*eyyn1[:nnx]+q1[:nnx],label="$t_y$ (C->N)")
  ax1.plot(-tx[:nnx],label="$t_x$ (CBF)")
  ax1.plot(-ty[:nnx],label="$t_y$ (CBF)")
  ax1.legend()
  ax1.set_title("Lower Boundary")

  ax2.plot(2*exyn1[(nnp-nnx):],label="$t_x$ (C->N)")
  ax2.plot(2*eyyn1[(nnp-nnx):]-q1[(nnp-nnx):],label="$t_y$ (C->N)")
  ax2.plot(tx[(nnp-nnx):],label="$t_x$ (CBF)")
  ax2.plot(ty[(nnp-nnx):],label="$t_y$ (CBF)")
  ax2.legend()
  ax2.set_title("Upper Boundary")


  #there's probably a better way of doing this
  #but I'm not at my best so here's a bit of a hack

  left_ty_cn = []
  right_ty_cn= []
  left_tx_cn = []
  right_tx_cn= []
  right_y = []
  left_y  = []

  for i in range(0,nnp):
    if x[i]<eps:
      left_tx_cn.append(2*exxn1[i]-q1[i])
      left_ty_cn.append(2*exyn1[i])
      left_y.append(y[i])
    if (Lx-x[i])<eps:
      right_tx_cn.append(2*exxn1[i]-q1[i])
      right_ty_cn.append(2*exyn1[i])
      right_y.append(y[i])

  right_y=np.array(right_y)
  left_y=np.array(left_y)
  left_tx_cn=np.array(left_tx_cn)
  right_tx_cn=np.array(right_tx_cn)
  left_ty_cn=np.array(left_ty_cn)
  right_ty_cn=np.array(right_ty_cn)

  ax3.plot(right_tx_cn,label="$t_x$ (C->N)")
  ax3.plot(right_ty_cn,label="$t_y$ (C->N)")
  ax3.plot(right_tx,label="$t_x$ (CBF)")
  ax3.plot(right_ty,label="$t_y$ (CBF)")
  ax3.legend()
  ax3.set_title("Right Boundary")

  ax4.plot(-left_tx_cn,label="$t_x$ (C->N)")
  ax4.plot(-left_ty_cn,label="$t_y$ (C->N)")
  ax4.plot(-left_tx,label="$t_x$ (CBF)")
  ax4.plot(-left_ty,label="$t_y$ (CBF)")
  ax4.legend()
  ax4.set_title("Left Boundary")

  fig.savefig("tractions_CN_CBF"+str(nelx)+".pdf")

  ##################################################
  # Plot the C->N tractions
  ##################################################

  fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

  #ax1 contains the lower tractions I guess

  ax1.plot(-2*exyn1[:nnx],label="$t_x$ (C->N)")
  ax1.plot(-2*eyyn1[:nnx]+q1[:nnx],label="$t_y$ (C->N)")
  ax1.plot(-sigma_xy(x[:nnx],0),label="$t_x$ analytical")
  ax1.plot(-sigma_yy(x[:nnx],0),label="$t_y$ analytical")
  ax1.legend()
  ax1.set_title("Lower Boundary")

  ax2.plot(2*exyn1[(nnp-nnx):],label="$t_x$ (C->N)")
  ax2.plot(2*eyyn1[(nnp-nnx):]-q1[(nnp-nnx):],label="$t_y$ (C->N)")
  ax2.plot(sigma_xy(x[(nnp-nnx):],1),label="$t_x$ analytical")
  ax2.plot(sigma_yy(x[(nnp-nnx):],1),label="$t_y$ analytical")
  ax2.legend()
  ax2.set_title("Upper Boundary")


  #there's probably a better way of doing this
  #but I'm not at my best so here's a bit of a hack

  left_ty = []
  right_ty= []
  left_tx = []
  right_tx= []
  right_y = []
  left_y  = []

  for i in range(0,nnp):
    if x[i]<eps:
      left_tx.append(2*exxn1[i]-q1[i])
      left_ty.append(2*exyn1[i])
      left_y.append(y[i])
    if (Lx-x[i])<eps:
      right_tx.append(2*exxn1[i]-q1[i])
      right_ty.append(2*exyn1[i])
      right_y.append(y[i])

  right_y=np.array(right_y)
  left_y=np.array(left_y)
  left_tx=np.array(left_tx)
  right_tx=np.array(right_tx)
  left_ty=np.array(left_ty)
  right_ty=np.array(right_ty)

  ax3.plot(right_tx,label="$t_x$ (C->N)")
  ax3.plot(right_ty,label="$t_y$ (C->N)")
  ax3.plot(sigma_xx(1,right_y),label="$t_x$ analytical")
  ax3.plot(sigma_yx(1,right_y),label="$t_y$ analytical")
  ax3.legend()
  ax3.set_title("Right Boundary")

  ax4.plot(-left_tx,label="$t_x$ (C->N)")
  ax4.plot(-left_ty,label="$t_y$ (C->N)")
  ax4.plot(-sigma_xx(0,left_y),label="$t_x$ analytical")
  ax4.plot(-sigma_yx(0,left_y),label="$t_y$ analytical")
  ax4.legend()
  ax4.set_title("Left Boundary")

  fig.savefig("tractions_CN"+str(nelx)+".pdf")







  #####################################################################
  # plot of solution
  #####################################################################

  filename = 'solution.vtu'
  vtufile=open(filename,"w")
  vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
  vtufile.write("<UnstructuredGrid> \n")
  vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
  #####
  vtufile.write("<Points> \n")
  vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
  vtufile.write("</DataArray>\n")
  vtufile.write("</Points> \n")
  #####
  vtufile.write("<CellData Scalars='scalars'>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%10e\n" % exx[iel])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%10e\n" % eyy[iel])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%10e\n" % exy[iel])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%10e\n" % p[iel])
  vtufile.write("</DataArray>\n")
  #--
  # vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
  # for iel in range (0,nel):
  #     vtufile.write("%10e\n" % sigmayy_el[iel])
  # vtufile.write("</DataArray>\n")

  vtufile.write("</CellData>\n")
  #####
  vtufile.write("<PointData Scalars='scalars'>\n")
  #--
  vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='tractions' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%20e %20e %20e \n" %(tx[i],ty[i],0.))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='q (C-N)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%20e \n" %q1[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='q (LS)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %q3[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='p (theoretical)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %pressure(x[i],y[i]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='eyy (C-N)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %eyyn1[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='eyy (LS)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %eyyn3[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='sxx (C-N)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %sxxn1[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='sxx (theoretical)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %sigma_xx(x[i],y[i]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='sxx (error)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %(sxxn1[i]-sigma_xx(x[i],y[i])))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='syy (C-N)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %syyn1[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='syy (theoretical)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %sigma_yy(x[i],y[i]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='syy (error)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %(syyn1[i]-sigma_yy(x[i],y[i])))
  vtufile.write("</DataArray>\n")
  #--
  #--
  vtufile.write("<DataArray type='Float32' Name='sxy (C-N)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %sxyn1[i])
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='sxy (theoretical)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %sigma_xy(x[i],y[i]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Float32' Name='sxy (error)' Format='ascii'> \n")
  for i in range(0,nnp):
      vtufile.write("%10e \n" %(sxyn1[i]-sigma_xy(x[i],y[i])))
  vtufile.write("</DataArray>\n")
  #--




  #--
  print(len(sxxn1),len(eyyn1))
  # vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
  # for i in range(0,nnp):
  #     vtufile.write("%10e \n" %rho[i])
  # vtufile.write("</DataArray>\n")
  #--
  vtufile.write("</PointData>\n")
  #####
  vtufile.write("<Cells>\n")
  #--
  vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
  for iel in range (0,nel):
      vtufile.write("%d \n" %((iel+1)*m))
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
  for iel in range (0,nel):
      vtufile.write("%d \n" %9)
  vtufile.write("</DataArray>\n")
  #--
  vtufile.write("</Cells>\n")
  #####
  vtufile.write("</Piece>\n")
  vtufile.write("</UnstructuredGrid>\n")
  vtufile.write("</VTKFile>\n")
  vtufile.close()

  ############################################################################
  ######## Calculate the L1 and L2 norms for ty on the all surfaces ##########
  ############################################################################

  left_elements_list=[]
  right_elements_list=[]
  for iel in range(nel):
    if (x[icon[0,iel]] < eps):
      left_elements_list.append(iel)
    if (abs(x[icon[1,iel]]-Lx) < eps):
      right_elements_list.append(iel)


  L1_norm_CBF=[0]*8
  L2_norm_CBF=[0]*8
  L1_norm_CN =[0]*8
  L2_norm_CN =[0]*8
  L1_norm_LS =[0]*8
  L2_norm_LS =[0]*8



  for iel in range(nel-nelx+1,nel-1): # upper norms ergo nodes 2 & 3
      for iq in [-1,1]:
          rq=iq/sqrt3

          N0=0.5*(1-rq)
          N1=0.5*(1+rq)
          wq=1*1

          xq=N0*x[icon[3,iel]]+N1*x[icon[2,iel]]
          tyq_CBF=N0*ty[icon[3,iel]]+N1*ty[icon[2,iel]]
          tyq_CN=N0*(2*viscosity*eyyn1[icon[3,iel]]-q1[icon[3,iel]])+N1*(2*viscosity*eyyn1[icon[2,iel]]-q1[icon[2,iel]])
          tyq_LS=N0*(2*viscosity*eyyn3[icon[3,iel]]-q3[icon[3,iel]])+N1*(2*viscosity*eyyn3[icon[2,iel]]-q3[icon[2,iel]])

          txq_CBF=N0*tx[icon[3,iel]]+N1*tx[icon[2,iel]]
          txq_CN=N0*(2*viscosity*exyn1[icon[3,iel]])+N1*(2*viscosity*exyn1[icon[2,iel]])
          txq_LS=N0*(2*viscosity*exyn3[icon[3,iel]])+N1*(2*viscosity*exyn3[icon[2,iel]])



          L1_norm_CBF[0]+=abs(tyq_CBF-sigma_yy(xq,1))*hx*wq
          L2_norm_CBF[0]+=(tyq_CBF-sigma_yy(xq,1))**2*hx*wq

          L1_norm_CN[0]+=abs(tyq_CN-sigma_yy(xq,1))*hx*wq
          L2_norm_CN[0]+=(tyq_CN-sigma_yy(xq,1))**2*hx*wq

          L1_norm_LS[0]+=abs(tyq_LS-sigma_yy(xq,1))*hx*wq
          L2_norm_LS[0]+=(tyq_LS-sigma_yy(xq,1))**2*hx*wq


          L1_norm_CBF[1]+=abs(txq_CBF-sigma_xy(xq,1))*hx*wq
          L2_norm_CBF[1]+=(txq_CBF-sigma_xy(xq,1))**2*hx*wq

          L1_norm_CN[1]+=abs(txq_CN-sigma_xy(xq,1))*hx*wq
          L2_norm_CN[1]+=(txq_CN-sigma_xy(xq,1))**2*hx*wq

          L1_norm_LS[1]+=abs(txq_LS-sigma_xy(xq,1))*hx*wq
          L2_norm_LS[1]+=(txq_LS-sigma_xy(xq,1))**2*hx*wq

  for iel in range(1,nelx-1): # lower norms ergo nodes 0 and 1
      for iq in [-1,1]:
          rq=iq/sqrt3

          N0=0.5*(1-rq)
          N1=0.5*(1+rq)
          wq=1*1

          xq=N0*x[icon[0,iel]]+N1*x[icon[1,iel]]
          tyq_CBF=N0*ty[icon[0,iel]]+N1*ty[icon[1,iel]]
          tyq_CN=-N0*(2*viscosity*eyyn1[icon[0,iel]]-q1[icon[0,iel]])-N1*(2*viscosity*eyyn1[icon[1,iel]]-q1[icon[1,iel]])
          tyq_LS=-N0*(2*viscosity*eyyn3[icon[0,iel]]-q3[icon[0,iel]])-N1*(2*viscosity*eyyn3[icon[1,iel]]-q3[icon[1,iel]])

          txq_CBF=N0*tx[icon[0,iel]]+N1*tx[icon[1,iel]]
          txq_CN=-N0*(2*viscosity*exyn1[icon[0,iel]])-N1*(2*viscosity*exyn1[icon[1,iel]])
          txq_LS=-N0*(2*viscosity*exyn3[icon[0,iel]])-N1*(2*viscosity*exyn3[icon[1,iel]])


          L1_norm_CBF[2]+=abs(tyq_CBF+sigma_yy(xq,0))*hx*wq
          L2_norm_CBF[2]+=(tyq_CBF+sigma_yy(xq,0))**2*hx*wq

          L1_norm_CN[2]+=abs(tyq_CN+sigma_yy(xq,0))*hx*wq
          L2_norm_CN[2]+=(tyq_CN+sigma_yy(xq,0))**2*hx*wq

          L1_norm_LS[2]+=abs(tyq_LS+sigma_yy(xq,0))*hx*wq
          L2_norm_LS[2]+=(tyq_LS+sigma_yy(xq,0))**2*hx*wq


          L1_norm_CBF[3]+=abs(txq_CBF+sigma_xy(xq,0))*hx*wq
          L2_norm_CBF[3]+=(txq_CBF+sigma_xy(xq,0))**2*hx*wq

          L1_norm_CN[3]+=abs(txq_CN+sigma_xy(xq,0))*hx*wq
          L2_norm_CN[3]+=(txq_CN+sigma_xy(xq,0))**2*hx*wq

          L1_norm_LS[3]+=abs(txq_LS+sigma_xy(xq,0))*hx*wq
          L2_norm_LS[3]+=(txq_LS+sigma_xy(xq,0))**2*hx*wq


  for i in range(1,len(left_elements_list)-1): #left norms ergo nodes 3 & 0
      iel=left_elements_list[i]
      for iq in [-1,1]:
          rq=iq/sqrt3

          N0=0.5*(1-rq)
          N1=0.5*(1+rq)
          wq=1*1

          yq=N0*y[icon[3,iel]]+N1*y[icon[0,iel]]
          txq_CBF=N0*tx[icon[3,iel]]+N1*tx[icon[0,iel]]
          txq_CN=-N0*(2*viscosity*exxn1[icon[3,iel]]-q1[icon[3,iel]])-N1*(2*viscosity*exxn1[icon[0,iel]]-q1[icon[0,iel]])
          txq_LS=-N0*(2*viscosity*exxn3[icon[3,iel]]-q3[icon[3,iel]])-N1*(2*viscosity*exxn3[icon[0,iel]]-q3[icon[0,iel]])

          tyq_CBF=N0*ty[icon[3,iel]]+N1*ty[icon[0,iel]]
          tyq_CN=-N0*(2*viscosity*exyn1[icon[3,iel]])-N1*(2*viscosity*exyn1[icon[0,iel]])
          tyq_LS=-N0*(2*viscosity*exyn3[icon[3,iel]])-N1*(2*viscosity*exyn3[icon[0,iel]])


          L1_norm_CBF[4]+=abs(tyq_CBF+sigma_xy(0,yq))*hx*wq
          L2_norm_CBF[4]+=(tyq_CBF+sigma_xy(0,yq))**2*hx*wq

          L1_norm_CN[4]+=abs(tyq_CN+sigma_xy(0,yq))*hx*wq
          L2_norm_CN[4]+=(tyq_CN+sigma_xy(0,yq))**2*hx*wq

          L1_norm_LS[4]+=abs(tyq_LS+sigma_xy(0,yq))*hx*wq
          L2_norm_LS[4]+=(tyq_LS+sigma_xy(0,yq))**2*hx*wq


          L1_norm_CBF[5]+=abs(txq_CBF+sigma_yy(0,yq))*hx*wq
          L2_norm_CBF[5]+=(txq_CBF+sigma_yy(0,yq))**2*hx*wq

          L1_norm_CN[5]+=abs(txq_CN+sigma_yy(0,yq))*hx*wq
          L2_norm_CN[5]+=(txq_CN+sigma_yy(0,yq))**2*hx*wq

          L1_norm_LS[5]+=abs(txq_LS+sigma_yy(0,yq))*hx*wq
          L2_norm_LS[5]+=(txq_LS+sigma_yy(0,yq))**2*hx*wq

  for i in range(1,len(right_elements_list)-1): #right norms ergo nodes 1 & 2
      iel=right_elements_list[i]
      for iq in [-1,1]:
          rq=iq/sqrt3

          N0=0.5*(1-rq)
          N1=0.5*(1+rq)
          wq=1*1

          yq=N0*y[icon[1,iel]]+N1*y[icon[2,iel]]
          txq_CBF=N0*tx[icon[1,iel]]+N1*tx[icon[2,iel]]
          txq_CN=N0*(2*viscosity*exxn1[icon[1,iel]]-q1[icon[1,iel]])+N1*(2*viscosity*exxn1[icon[2,iel]]-q1[icon[2,iel]])
          txq_LS=N0*(2*viscosity*exxn3[icon[1,iel]]-q3[icon[1,iel]])+N1*(2*viscosity*exxn3[icon[2,iel]]-q3[icon[2,iel]])

          tyq_CBF=N0*ty[icon[1,iel]]+N1*ty[icon[2,iel]]
          tyq_CN=N0*(2*viscosity*exyn1[icon[1,iel]])+N1*(2*viscosity*exyn1[icon[2,iel]])
          tyq_LS=N0*(2*viscosity*exyn3[icon[1,iel]])+N1*(2*viscosity*exyn3[icon[2,iel]])


          L1_norm_CBF[6]+=abs(tyq_CBF-sigma_xy(1,yq))*hx*wq
          L2_norm_CBF[6]+=(tyq_CBF-sigma_xy(1,yq))**2*hx*wq

          L1_norm_CN[6]+=abs(tyq_CN-sigma_xy(1,yq))*hx*wq
          L2_norm_CN[6]+=(tyq_CN-sigma_xy(1,yq))**2*hx*wq

          L1_norm_LS[6]+=abs(tyq_LS-sigma_xy(1,yq))*hx*wq
          L2_norm_LS[6]+=(tyq_LS-sigma_xy(1,yq))**2*hx*wq


          L1_norm_CBF[7]+=abs(txq_CBF-sigma_yy(1,yq))*hx*wq
          L2_norm_CBF[7]+=(txq_CBF-sigma_yy(1,yq))**2*hx*wq

          L1_norm_CN[7]+=abs(txq_CN-sigma_yy(1,yq))*hx*wq
          L2_norm_CN[7]+=(txq_CN-sigma_yy(1,yq))**2*hx*wq

          L1_norm_LS[7]+=abs(txq_LS-sigma_yy(1,yq))*hx*wq
          L2_norm_LS[7]+=(txq_LS-sigma_yy(1,yq))**2*hx*wq




  #print(L1_norm_list_CBF)
  for i in range(8):
    L1_norm_list_CBF[i].append(L1_norm_CBF[i])
    L2_norm_list_CBF[i].append(np.sqrt(L2_norm_CBF[i]))
    L1_norm_list_CN[i].append(L1_norm_CN[i])
    L2_norm_list_CN[i].append(np.sqrt(L2_norm_CN[i]))
    L1_norm_list_LS[i].append(L1_norm_LS[i])
    L2_norm_list_LS[i].append(np.sqrt(L2_norm_LS[i]))
    # print(L1_norm_CBF[i])
    # print(L1_norm_list_CBF[i])


#print(L1_norm_list_CBF)

hx_list=1/(np.array(nelx_list)+1)
log_hx=np.log(np.abs(hx_list))

fig,ax=plt.subplots(4,2,figsize=(20,40))

for i in range(8):
  print(i)
  print(i%2)
  print(int(np.floor(i/2)))
  #ax.plot(log_hx,np.log(np.abs(corner_error_list)),label="corner CBF")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_CBF[i]),label="L1 CBF")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_CBF[i]),label="L2 CBF")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_CN[i]),label="L1 CN")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_CN[i]),label="L2 CN")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_LS[i]),label="L1 LS")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_LS[i]),label="L2 LS")
  ax[int(np.floor(i/2))][i%2].grid()
  ax[int(np.floor(i/2))][i%2].legend()
  ax[int(np.floor(i/2))][i%2].set_xlabel("log(1/nnx)")
  ax[int(np.floor(i/2))][i%2].set_ylabel("Norm Magnitude")
  #ax[i%2][np.floor(i/2)].set_title("Convergence Rates for Traction Norms")

ax[0][0].set_title("Upper surface $t_y$")
ax[0][1].set_title("Upper surface $t_x$")
ax[1][0].set_title("Lower surface $t_y$")
ax[1][1].set_title("Lower surface $t_x$")
ax[2][0].set_title("Left surface $t_x$")
ax[2][1].set_title("Left surface $t_y$")
ax[3][0].set_title("Right surface $t_x$")
ax[3][1].set_title("Right surface $t_y$")

fig.savefig("convergence.pdf")


######Now do the regressions

L1_norm_CBF_regression=np.zeros(8,dtype=np.float64)
L1_norm_CN_regression=np.zeros(8,dtype=np.float64)
L1_norm_LS_regression=np.zeros(8,dtype=np.float64)
L2_norm_CBF_regression=np.zeros(8,dtype=np.float64)
L2_norm_CN_regression=np.zeros(8,dtype=np.float64)
L2_norm_LS_regression=np.zeros(8,dtype=np.float64)

for i in range(8):  
  print(i)
  L1_norm_CBF_regression[i] = get_regression(hx_list,L1_norm_list_CBF[i])
  L1_norm_CN_regression[i]  = get_regression(hx_list,L1_norm_list_CN[i])
  L1_norm_LS_regression[i]  = get_regression(hx_list,L1_norm_list_LS[i])

  L2_norm_CBF_regression[i] = get_regression(hx_list,L2_norm_list_CBF[i])
  L2_norm_CN_regression[i]  = get_regression(hx_list,L2_norm_list_CN[i])
  L2_norm_LS_regression[i]  = get_regression(hx_list,L2_norm_list_LS[i])


  print("L1 norm convergence rates")
  print("CBF:" + str(L1_norm_CBF_regression[i]))
  print("CN :" + str(L1_norm_CN_regression[i]))
  print("LS :" + str(L1_norm_LS_regression[i]))

  print("L2 norm convergence rates")
  print("CBF:" + str(L2_norm_CBF_regression[i]))
  print("CN :" + str(L2_norm_CN_regression[i]))
  print("LS :" + str(L2_norm_LS_regression[i]))

  L1_CBF_CN_difference = get_regression_x_intercept_difference(hx_list,L1_norm_list_CBF[i],L1_norm_list_CN[i])
  L1_CBF_LS_difference = get_regression_x_intercept_difference(hx_list,L1_norm_list_CBF[i],L1_norm_list_LS[i])
  L1_LS_CN_difference  = get_regression_x_intercept_difference(hx_list,L1_norm_list_LS[i],L1_norm_list_CN[i])

  L2_CBF_CN_difference = get_regression_x_intercept_difference(hx_list,L2_norm_list_CBF[i],L2_norm_list_CN[i])
  L2_CBF_LS_difference = get_regression_x_intercept_difference(hx_list,L2_norm_list_CBF[i],L2_norm_list_LS[i])
  L2_LS_CN_difference  = get_regression_x_intercept_difference(hx_list,L2_norm_list_LS[i],L2_norm_list_CN[i])

  print("L1 LS CN ",L1_LS_CN_difference)
  print("L1 CN CBF ",L1_CBF_CN_difference)
  print("L1 CBF LS ",L1_CBF_LS_difference)

  print("L2 LS CN ",L2_LS_CN_difference)
  print("L2 CN CBF ",L2_CBF_CN_difference)
  print("L2 CBF LS ",L2_CBF_LS_difference)


filename="table_convergences_L1"
file=open(filename,'w')
file.write("\\begin{table}\n")
file.write("\label{table:q2q1}\n")
file.write("\caption{$L_1$ norm convergences for the $Q_1P_0$ Donea and Huerta benchmark.}\n")
file.write("\\begin{center}\n")
file.write("\\begin{tabular}{| c | c c | c c | c c |} \n")
file.write("\hline\n")
file.write("Method & \multicolumn{2}{|c|}{Upper} & \multicolumn{2}{|c|}{Lower} & \multicolumn{2}{|c|}{Sides} \\\ \hline \n")
file.write("& $t_x$ & $t_y$ & $t_x$ & $t_y$ & $t_x$ & $t_y$  \\\ \hline\n")
file.write("LS   &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L1_norm_LS_regression[1],L1_norm_LS_regression[0],L1_norm_LS_regression[3],L1_norm_LS_regression[2],L1_norm_LS_regression[5],L1_norm_LS_regression[4]))
file.write("CN   &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L1_norm_CN_regression[1],L1_norm_CN_regression[0],L1_norm_CN_regression[3],L1_norm_CN_regression[2],L1_norm_CN_regression[5],L1_norm_CN_regression[4]))
file.write("CBF  &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L1_norm_CBF_regression[1],L1_norm_CBF_regression[0],L1_norm_CBF_regression[3],L1_norm_CBF_regression[2],L1_norm_CBF_regression[5],L1_norm_CBF_regression[4]))
file.write("\end{tabular}\n")
file.write("\end{center}\n")
file.write("\end{table}\n")
file.close()

filename="table_convergences_L2"
file=open(filename,'w')
file.write("\\begin{table}\n")
file.write("\label{table:q2q1}\n")
file.write("\caption{$L_2$ norm convergences for the $Q_1P_0$ Donea and Huerta benchmark.}\n")
file.write("\\begin{center}\n")
file.write("\\begin{tabular}{| c | c c | c c | c c |} \n")
file.write("\hline\n")
file.write("Method & \multicolumn{2}{|c|}{Upper} & \multicolumn{2}{|c|}{Lower} & \multicolumn{2}{|c|}{Sides} \\\ \hline\n")
file.write("& $t_x$ & $t_y$ & $t_x$ & $t_y$ & $t_x$ & $t_y$  \\\ \hline\n")
file.write("LS   &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L2_norm_LS_regression[1],L2_norm_LS_regression[0],L2_norm_LS_regression[3],L2_norm_LS_regression[2],L2_norm_LS_regression[5],L2_norm_LS_regression[4]))
file.write("CN   &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L2_norm_CN_regression[1],L2_norm_CN_regression[0],L2_norm_CN_regression[3],L2_norm_CN_regression[2],L2_norm_CN_regression[5],L2_norm_CN_regression[4]))
file.write("CBF  &  {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(
L2_norm_CBF_regression[1],L2_norm_CBF_regression[0],L2_norm_CBF_regression[3],L2_norm_CBF_regression[2],L2_norm_CBF_regression[5],L2_norm_CBF_regression[4]))
file.write("\end{tabular}\n")
file.write("\end{center}\n")
file.write("\end{table}\n")
file.close()





print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


