import numpy as np
import math as math
import sys as sys
#import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time
import random

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

def uth(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def vth(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pth(x,y):
    val=x*(1.-x)-1./6.
    return val

def exxth(x,y):
    val=4*x*y*(1-x)*(1-2*x)*(1-3*y+2*y**2)
    return val

def eyyth(x,y):
    val=-4*x*y*(1-y)*(1-2*y)*(1-3*x+2*x**2)
    return val

def exyth(x,y):
    val=2*x**2*(1-x)**2*(1-6*y+6*y**2) /2. \
       -2*y**2*(1-y)**2*(1-6*x+6*x**2) /2.
    return val

def srth(x,y):
    valxx=4*x*y*(1-x)*(1-2*x)*(1-3*y+2*y**2)
    valyy=-4*x*y*(1-y)*(1-2*y)*(1-3*x+2*x**2)
    valxy=2*x**2*(1-x)**2*(1-6*y+6*y**2) /2. \
         -2*y**2*(1-y)**2*(1-6*x+6*x**2) /2.
    val=np.sqrt(0.5*valxx**2+0.5*valyy**2+valxy**2)  
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

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
print("variable declaration")

m=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

viscosity=1.  # dynamic viscosity \mu

NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10
sqrt3=np.sqrt(3.)

hx=Lx/nelx
hy=Ly/nely

random_grid=False
penalty=1.e7  # penalty coefficient value
sparse=False
pnormalise=True

eta_ref=1.
pressure_scaling=eta_ref/Lx

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

if random_grid:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           if i>0 and j>0 and i<nnx-1 and j<nny-1:
              dx=2.*(random.random()-0.5)*hx/20
              dy=2.*(random.random()-0.5)*hy/20
           else:
              dx=0
              dy=0
           x[counter]=i*Lx/float(nelx)+dx
           y[counter]=j*Ly/float(nely)+dy
           counter += 1
else:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*Lx/float(nelx)
           y[counter]=j*Ly/float(nely)
           counter += 1

np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

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

start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
on_bd=np.zeros((nnp,4),dtype=np.bool)  # boundary indicator
 
for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,0]=True
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,1]=True
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,2]=True
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,3]=True

print("setup: boundary conditions: %.3f s" % (time.time() - start))
#################################################################
# build FE matrix
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
            a_el += b_mat.T.dot((c_mat).dot(b_mat))*viscosity*wq*jcob

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

#####################################################################
# compute nodal strain rate - method 1 : center to node
#####################################################################

p1=np.zeros(nnp,dtype=np.float64)  
exx1=np.zeros(nnp,dtype=np.float64)  
eyy1=np.zeros(nnp,dtype=np.float64)  
exy1=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):

    exx1[icon[0,iel]]+=exx[iel]
    exx1[icon[1,iel]]+=exx[iel]
    exx1[icon[2,iel]]+=exx[iel]
    exx1[icon[3,iel]]+=exx[iel]

    eyy1[icon[0,iel]]+=eyy[iel]
    eyy1[icon[1,iel]]+=eyy[iel]
    eyy1[icon[2,iel]]+=eyy[iel]
    eyy1[icon[3,iel]]+=eyy[iel]

    exy1[icon[0,iel]]+=exy[iel]
    exy1[icon[1,iel]]+=exy[iel]
    exy1[icon[2,iel]]+=exy[iel]
    exy1[icon[3,iel]]+=exy[iel]

    p1[icon[0,iel]]+=p[iel]
    p1[icon[1,iel]]+=p[iel]
    p1[icon[2,iel]]+=p[iel]
    p1[icon[3,iel]]+=p[iel]

    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1

exx1/=count
eyy1/=count
exy1/=count
p1/=count

print("     -> exx1 (m,M) %.4f %.4f " %(np.min(exx1),np.max(exx1)))
print("     -> eyy1 (m,M) %.4f %.4f " %(np.min(eyy1),np.max(eyy1)))
print("     -> exy1 (m,M) %.4f %.4f " %(np.min(exy1),np.max(exy1)))

np.savetxt('srn_1.ascii',np.array([x,y,exx1,eyy1,exy1]).T,header='# x,y,exx1,eyy1,exy1')
np.savetxt('p_1.ascii',np.array([x,y,p1]).T,header='# x,y,p')

#####################################################################
# compute nodal strain rate - method 2 : corners to node
#####################################################################

exx2=np.zeros(nnp,dtype=np.float64)  
eyy2=np.zeros(nnp,dtype=np.float64)  
exy2=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):

    # lower left
    rq=-1.
    sq=-1.
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[0,iel]]+=exx_rs
    eyy2[icon[0,iel]]+=eyy_rs
    exy2[icon[0,iel]]+=exy_rs
    count[icon[0,iel]]+=1

    # lower right
    rq=+1.
    sq=-1.
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[1,iel]]+=exx_rs
    eyy2[icon[1,iel]]+=eyy_rs
    exy2[icon[1,iel]]+=exy_rs
    count[icon[1,iel]]+=1

    # upper right
    rq=+1.
    sq=+1.
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[2,iel]]+=exx_rs
    eyy2[icon[2,iel]]+=eyy_rs
    exy2[icon[2,iel]]+=exy_rs
    count[icon[2,iel]]+=1

    # upper left
    rq=-1.
    sq=+1.
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[3,iel]]+=exx_rs
    eyy2[icon[3,iel]]+=eyy_rs
    exy2[icon[3,iel]]+=exy_rs
    count[icon[3,iel]]+=1

exx2/=count
eyy2/=count
exy2/=count

print("     -> exx2 (m,M) %.4f %.4f " %(np.min(exx2),np.max(exx2)))
print("     -> eyy2 (m,M) %.4f %.4f " %(np.min(eyy2),np.max(eyy2)))
print("     -> exy2 (m,M) %.4f %.4f " %(np.min(exy2),np.max(exy2)))

np.savetxt('srn_2.ascii',np.array([x,y,exx2,eyy2,exy2]).T,header='# x,y,exx2,eyy2,exy2')

#####################################################################
# compute nodal strain rate - method 3: Superconvergence Patch Recovery 
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

p3=np.zeros(nnp,dtype=np.float64)  
exx3=np.zeros(nnp,dtype=np.float64)  
eyy3=np.zeros(nnp,dtype=np.float64)  
exy3=np.zeros(nnp,dtype=np.float64)  

AA = np.zeros((4,4),dtype=np.float64) 
BBxx = np.zeros(4,dtype=np.float64) 
BByy = np.zeros(4,dtype=np.float64) 
BBxy = np.zeros(4,dtype=np.float64) 
BBp  = np.zeros(4,dtype=np.float64) 

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

           # print(np.linalg.cond(AA))
           # print("\n")


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

           BBp[0]=p[iel0] 
           BBp[1]=p[iel1] 
           BBp[2]=p[iel2] 
           BBp[3]=p[iel3] 
           solp=sps.linalg.spsolve(sps.csr_matrix(AA),BBp)
           
           # node 4 of patch
           ip=icon[2,iel0] 
           exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
           eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
           exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
           p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # node 1 of patch
           ip=icon[1,iel0] 
           if on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # node 3 of patch
           ip=icon[3,iel0] 
           if on_bd[ip,0]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # node 5 of patch
           ip=icon[2,iel1] 
           if on_bd[ip,1]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # node 7 of patch
           ip=icon[3,iel2] 
           if on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # lower left corner of domain
           ip=icon[0,iel0] 
           if on_bd[ip,0] and on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # lower right corner of domain
           ip=icon[1,iel1] 
           if on_bd[ip,1] and on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # upper right corner of domain
           ip=icon[2,iel2] 
           if on_bd[ip,1] and on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

           # lower right corner of domain
           ip=icon[3,iel3] 
           if on_bd[ip,0] and on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]
              p3[ip]=solp[0]+solp[1]*x[ip]+solp[2]*y[ip]+solp[3]*x[ip]*y[ip]

        counter+=1

print("     -> exx3 (m,M) %.4f %.4f " %(np.min(exx3),np.max(exx3)))
print("     -> eyy3 (m,M) %.4f %.4f " %(np.min(eyy3),np.max(eyy3)))
print("     -> exy3 (m,M) %.4f %.4f " %(np.min(exy3),np.max(exy3)))

np.savetxt('srn_3.ascii',np.array([x,y,exx3,eyy3,exy3]).T,header='# x,y,exx3,eyy3,exy3')
np.savetxt('p_3.ascii',np.array([x,y,p3]).T,header='# x,y,p')

#####################################################################
# compute nodal strain rate - method 4: global 
#####################################################################

reduced=False

p4=np.zeros(nnp,dtype=np.float64)  
exx4=np.zeros(nnp,dtype=np.float64)  
eyy4=np.zeros(nnp,dtype=np.float64)  
exy4=np.zeros(nnp,dtype=np.float64)  

A_mat=np.zeros((nnp,nnp),dtype=np.float64) # Q1 mass matrix
rhs_xx=np.zeros(nnp,dtype=np.float64)      # rhs
rhs_yy=np.zeros(nnp,dtype=np.float64)      # rhs
rhs_xy=np.zeros(nnp,dtype=np.float64)      # rhs
rhs_p=np.zeros(nnp,dtype=np.float64)       # rhs
 
for iel in range(0, nel):

    # set arrays to 0 every loop
    fp_el =np.zeros(m,dtype=np.float64)
    fxx_el =np.zeros(m,dtype=np.float64)
    fyy_el =np.zeros(m,dtype=np.float64)
    fxy_el =np.zeros(m,dtype=np.float64)
    M_el =np.zeros((m,m),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            exxq=0.0
            eyyq=0.0
            exyq=0.0
            for k in range(0, m):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                exxq += dNdx[k]*u[icon[k,iel]]
                eyyq += dNdy[k]*v[icon[k,iel]]
                exyq += (dNdx[k]*v[icon[k,iel]]+dNdy[k]*u[icon[k,iel]])*0.5

            for i in range(0,m):
                for j in range(0,m):
                    M_el[i,j]+=N[i]*N[j]*wq*jcob
                if not reduced:
                   fxx_el[i]+=N[i]*exxq*wq*jcob
                   fyy_el[i]+=N[i]*eyyq*wq*jcob
                   fxy_el[i]+=N[i]*exyq*wq*jcob
                   fp_el[i]+=N[i]*p[iel]*wq*jcob

    if reduced:
       rq=0.
       sq=0.
       wq=2.*2.
       N[0:m]=NNV(rq,sq)
       dNdr[0:m]=dNNVdr(rq,sq)
       dNds[0:m]=dNNVds(rq,sq)
       jcb = np.zeros((2,2),dtype=np.float64)
       for k in range(0,m):
           jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
           jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
           jcb[1,0]+=dNds[k]*x[icon[k,iel]]
           jcb[1,1]+=dNds[k]*y[icon[k,iel]]
       jcob=np.linalg.det(jcb)
       jcbi=np.linalg.inv(jcb)
       exxq=0.0
       eyyq=0.0
       exyq=0.0
       for k in range(0, m):
           dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
           dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
           exxq += dNdx[k]*u[icon[k,iel]]
           eyyq += dNdy[k]*v[icon[k,iel]]
           exyq += (dNdx[k]*v[icon[k,iel]]+dNdy[k]*u[icon[k,iel]])*0.5
       for i in range(0,m):
           fxx_el[i]+=N[i]*exxq*wq*jcob
           fyy_el[i]+=N[i]*eyyq*wq*jcob
           fxy_el[i]+=N[i]*exyq*wq*jcob
           fp_el[i]+=N[i]*p[iel]*wq*jcob

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        ik=icon[k1,iel]
        for k2 in range(0,m):
            jk=icon[k2,iel]
            A_mat[ik,jk]+=M_el[k1,k2]
        rhs_xx[ik]+=fxx_el[k1]
        rhs_yy[ik]+=fyy_el[k1]
        rhs_xy[ik]+=fxy_el[k1]
        rhs_p[ik]+=fp_el[k1]

exx4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xx)
eyy4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_yy)
exy4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xy)
p4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_p)

print("     -> exx4 (m,M) %.4f %.4f " %(np.min(exx4),np.max(exx4)))
print("     -> eyy4 (m,M) %.4f %.4f " %(np.min(eyy4),np.max(eyy4)))
print("     -> exy4 (m,M) %.4f %.4f " %(np.min(exy4),np.max(exy4)))

np.savetxt('srn_4.ascii',np.array([x,y,exx4,eyy4,exy4]).T,header='# x,y,exx4,eyy4,exy4')
np.savetxt('p_4.ascii',np.array([x,y,p4]).T,header='# x,y,p')


######################################################################
# Method 5: Finite Nodal Derivatives
######################################################################

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

#p3=np.zeros(nnp,dtype=np.float64)  
exx5=np.zeros(nnp,dtype=np.float64)  
eyy5=np.zeros(nnp,dtype=np.float64)  
exy5=np.zeros(nnp,dtype=np.float64)  

vx_reconstructed=np.zeros(nnp,dtype=np.float64)
vy_reconstructed=np.zeros(nnp,dtype=np.float64)
p_reconstructed =np.zeros(nnp,dtype=np.float64)


counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        if i<nelx-1 and j<nely-1:
           iel0=counter
           iel1=counter+1
           iel2=counter+nelx+1
           iel3=counter+nelx

           node = icon[2,iel0]

           element_list=[iel0,iel1,iel2,iel3]

           rhs_x = 0
           rhs_y = 0
           lhs_x = 0
           lhs_y = 0
           lhs_xy= 0
           gel_x = 0
           gel_y = 0

           lhs=np.zeros((3,2),dtype=np.float64)
           rhs=np.zeros(3,dtype=np.float64)

           for iel_patch in range(4):


              f_el =np.zeros((m*ndofV),dtype=np.float64)
              K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
              # G_el=np.zeros((m*ndofV,1),dtype=np.float64)
              # h_el=np.zeros((1,1),dtype=np.float64)


              # integrate viscous term at 4 quadrature points
              iel=element_list[iel_patch]
              for iq in [-1, 1]:
                for jq in [-1, 1]:



                    # position & weight of quad. point
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    wq=1.*1.

                    # calculate shape functions
                    N[0:m]=NNV(rq,sq)
                    dNdr[0:m]=dNNVdr(rq,sq)
                    dNds[0:m]=dNNVds(rq,sq)

                    # calculate jacobian matrix
                    jcb = np.zeros((2,2),dtype=np.float64)
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
                    K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

                    # compute elemental rhs vector
                    for i in range(0, m):
                        f_el[ndofV*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                        f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq)



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
              K_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob


              if i==0:
                rhs_x += f_el[4]
                rhs_y += f_el[5]

                lhs_x += K_el[4,4]
                lhs_y += K_el[5,5]
                lhs_xy+= K_el[4,5]


                for k in [0,1,3]:
                  rhs_x -= K_el[4,2*k]  *u[icon[k,iel]]
                  rhs_y -= K_el[5,2*k+1]*v[icon[k,iel]]
                  


              if i==1:
                rhs_x += f_el[6]
                rhs_y += f_el[7]

                for k in [0,1,2]:
                  rhs_x -= K_el[6,2*k]  *u[icon[k,iel]]
                  rhs_y -= K_el[7,2*k+1]*v[icon[k,iel]]


                lhs_x += K_el[6,6]
                lhs_y += K_el[7,7]
                lhs_xy+= K_el[6,7]



              if i==2:
                rhs_x += f_el[0]
                rhs_y += f_el[1]

                for k in [1,2,3]:
                  rhs_x -= K_el[0,2*k]  *u[icon[k,iel]]
                  rhs_y -= K_el[1,2*k+1]*v[icon[k,iel]]

                lhs_x += K_el[0,0]
                lhs_y += K_el[1,1]
                lhx_xy+= K_el[0,1]


              if i==3:
                rhs_x += f_el[2]
                rhs_y += f_el[3]

                for k in [0,2,3]:
                  rhs_x -= K_el[2,2*k]  *u[icon[k,iel]]
                  rhs_y -= K_el[3,2*k+1]*v[icon[k,iel]]

                lhs_x += K_el[2,2]
                lhs_y += K_el[3,3]
                lhs_xy+= K_el[2,3]


           lhs[0,0]=lhs_x
           lhs[1,0]=lhs_xy
           lhs[0,1]=lhs_xy
           lhs[1,1]=lhs_y


           rhs[0]=rhs_x
           rhs[1]=rhs_y
           #print("test")

           #mrrgl=np.matmul(np.transpose(lhs),lhs)
           #print("test")


           #grrgl=np.matmul(np.transpose(lhs),rhs)
           #print("test")

           #sol=np.linalg.solve(np.matmul(np.transpose(lhs),lhs),np.matmul(np.transpose(lhs),rhs))

           #print("test")
           # vx_reconstructed[node] = sol[0]
           # vy_reconstructed[node] = sol[1]

           vx_reconstructed[node] = rhs_x/lhs_x
           vy_reconstructed[node] = rhs_y/lhs_y

        counter +=1


div_v_recon=np.zeros(nel,dtype=np.float64)

for iel in range(0, nel):
    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
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
            for k in range(0, m):

                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

                div_v_recon[iel] +=dNdy[k]*vy_reconstructed[icon[k,iel]]+dNdx[k]*vx_reconstructed[icon[k,iel]]


    for k in range(m):
      for i in range(4):
        if on_bd[icon[k,iel],i]:
          div_v_recon[iel]=0



######################################################################
# compute error
######################################################################
start = time.time()

errv=0.
errp=0. ; errp1=0 ; errp3=0. ; errp4=0.
errexx0=0. ; errexx1=0. ; errexx2=0. ; errexx3=0. ; errexx4=0. ; errexx5=0.
erreyy0=0. ; erreyy1=0. ; erreyy2=0. ; erreyy3=0. ; erreyy4=0. ; erreyy5=0.
errexy0=0. ; errexy1=0. ; errexy2=0. ; errexy3=0. ; errexy4=0. ; errexy5=0.

for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0, m):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            p1q=0. ; p3q=0. ; p4q=0.
            exx1q=0. ; exx2q=0. ; exx3q=0. ; exx4q=0. ; exx5q=0.
            eyy1q=0. ; eyy2q=0. ; eyy3q=0. ; eyy4q=0. ; eyy5q=0.
            exy1q=0. ; exy2q=0. ; exy3q=0. ; exy4q=0. ; exy5q=0.
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
                exx1q+=N[k]*exx1[icon[k,iel]]
                exx2q+=N[k]*exx2[icon[k,iel]]
                exx3q+=N[k]*exx3[icon[k,iel]]
                exx4q+=N[k]*exx4[icon[k,iel]]
                exx5q+=dNdx[k]*u[icon[k,iel]]
                eyy1q+=N[k]*eyy1[icon[k,iel]]
                eyy2q+=N[k]*eyy2[icon[k,iel]]
                eyy3q+=N[k]*eyy3[icon[k,iel]]
                eyy4q+=N[k]*eyy4[icon[k,iel]]
                eyy5q+=dNdy[k]*v[icon[k,iel]]
                exy1q+=N[k]*exy1[icon[k,iel]]
                exy2q+=N[k]*exy2[icon[k,iel]]
                exy3q+=N[k]*exy3[icon[k,iel]]
                exy4q+=N[k]*exy4[icon[k,iel]]
                exy5q+=dNdx[k]*v[icon[k,iel]]*0.5\
                      +dNdy[k]*u[icon[k,iel]]*0.5
                p1q  +=N[k]*p1[icon[k,iel]]
                p3q  +=N[k]*p3[icon[k,iel]]
                p4q  +=N[k]*p4[icon[k,iel]]
            exx0q=exx[iel]
            eyy0q=eyy[iel]
            exy0q=exy[iel]
            errv+=((uq-uth(xq,yq))**2+(vq-vth(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pth(xq,yq))**2*weightq*jcob
            errexx0+=(exx0q-exxth(xq,yq))**2*weightq*jcob
            errexx1+=(exx1q-exxth(xq,yq))**2*weightq*jcob
            errexx2+=(exx2q-exxth(xq,yq))**2*weightq*jcob
            errexx3+=(exx3q-exxth(xq,yq))**2*weightq*jcob
            errexx4+=(exx4q-exxth(xq,yq))**2*weightq*jcob
            errexx5+=(exx5q-exxth(xq,yq))**2*weightq*jcob
            erreyy0+=(eyy0q-eyyth(xq,yq))**2*weightq*jcob
            erreyy1+=(eyy1q-eyyth(xq,yq))**2*weightq*jcob
            erreyy2+=(eyy2q-eyyth(xq,yq))**2*weightq*jcob
            erreyy3+=(eyy3q-eyyth(xq,yq))**2*weightq*jcob
            erreyy4+=(eyy4q-eyyth(xq,yq))**2*weightq*jcob
            erreyy5+=(eyy5q-eyyth(xq,yq))**2*weightq*jcob
            errexy0+=(exy0q-exyth(xq,yq))**2*weightq*jcob
            errexy1+=(exy1q-exyth(xq,yq))**2*weightq*jcob
            errexy2+=(exy2q-exyth(xq,yq))**2*weightq*jcob
            errexy3+=(exy3q-exyth(xq,yq))**2*weightq*jcob
            errexy4+=(exy4q-exyth(xq,yq))**2*weightq*jcob
            errexy5+=(exy5q-exyth(xq,yq))**2*weightq*jcob
            errp1  +=(p1q-pth(xq,yq))**2*weightq*jcob
            errp3  +=(p3q-pth(xq,yq))**2*weightq*jcob
            errp4  +=(p4q-pth(xq,yq))**2*weightq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errp1=np.sqrt(errp1)
errp3=np.sqrt(errp3)
errp4=np.sqrt(errp4)

errexx0=np.sqrt(errexx0)
errexx1=np.sqrt(errexx1)
errexx2=np.sqrt(errexx2)
errexx3=np.sqrt(errexx3)
errexx4=np.sqrt(errexx4)
errexx5=np.sqrt(errexx5)

erreyy0=np.sqrt(erreyy0)
erreyy1=np.sqrt(erreyy1)
erreyy2=np.sqrt(erreyy2)
erreyy3=np.sqrt(erreyy3)
erreyy4=np.sqrt(erreyy4)
erreyy5=np.sqrt(erreyy5)

errexy0=np.sqrt(errexy0)
errexy1=np.sqrt(errexy1)
errexy2=np.sqrt(errexy2)
errexy3=np.sqrt(errexy3)
errexy4=np.sqrt(errexy4)
errexy5=np.sqrt(errexy5)

print("     -> nel= %6d ; errv= %.8e ; errp= %.8e " %(nel,errv,errp))
print("     -> nel= %6d ; errexx0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,errexx0,errexx1,errexx2,errexx3,errexx4,errexx5))
print("     -> nel= %6d ; erreyy0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,erreyy0,erreyy1,erreyy2,erreyy3,erreyy4,erreyy5))
print("     -> nel= %6d ; errexy0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,errexy0,errexy1,errexy2,errexy3,errexy4,errexy5))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:

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
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (exx[iel]-exxth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eyy[iel]-eyyth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (exy[iel]-exyth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v recon' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % div_v_recon[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (np.sqrt(exx[iel]**2+eyy[iel]**2+2*exy[iel]**2)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")

       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='p1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % p1[i])
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='p3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % p3[i])
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='p4' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % p4[i])
       vtufile.write("</DataArray>\n")


       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx1[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy1[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy1[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")

       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx2[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy2[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy2[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")

       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx3[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy3[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy3[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx4' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx4[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy4' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy4[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy4' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy4[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx4 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx4[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy4 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy4[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy4 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy4[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vx recon' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (vx_reconstructed[i]))#-u[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vy recon' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (vy_reconstructed[i]))#-v[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vx recon err' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (vx_reconstructed[i]-u[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vy recon err' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (vy_reconstructed[i]-v[i]))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sigma xx' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (2*exxth(x[i],y[i])-p1[i]))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sigma yy' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (2*eyyth(x[i],y[i])-p1[i]))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sigma xy' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (2*exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
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
           vtufile.write("%d \n" %((iel+1)*4))
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

print("generate vtu: %.3f s" % (time.time() - start))

# #####################################################################
# # using markers  
# #####################################################################
# start = time.time()

# if random_grid:
#    print ("random grid is on!!")

# nmarker=000

# xm = np.zeros(nmarker,dtype=np.float64)  # x coordinates
# ym = np.zeros(nmarker,dtype=np.float64)  # y coordinates

# exx0m = np.zeros(nmarker,dtype=np.float64) # elemental  
# exx1m = np.zeros(nmarker,dtype=np.float64) # nodal method 1 
# exx2m = np.zeros(nmarker,dtype=np.float64) # nodal method 2 
# exx3m = np.zeros(nmarker,dtype=np.float64) # nodal method 3 
# exx4m = np.zeros(nmarker,dtype=np.float64) # nodal method 4 
# exx5m = np.zeros(nmarker,dtype=np.float64) # shape fct derivatives

# eyy0m = np.zeros(nmarker,dtype=np.float64)  
# eyy1m = np.zeros(nmarker,dtype=np.float64)  
# eyy2m = np.zeros(nmarker,dtype=np.float64)  
# eyy3m = np.zeros(nmarker,dtype=np.float64)  
# eyy4m = np.zeros(nmarker,dtype=np.float64)  
# eyy5m = np.zeros(nmarker,dtype=np.float64)  

# exy0m = np.zeros(nmarker,dtype=np.float64)  
# exy1m = np.zeros(nmarker,dtype=np.float64)  
# exy2m = np.zeros(nmarker,dtype=np.float64)  
# exy3m = np.zeros(nmarker,dtype=np.float64)  
# exy4m = np.zeros(nmarker,dtype=np.float64)  
# exy5m = np.zeros(nmarker,dtype=np.float64)  

# sr0m = np.zeros(nmarker,dtype=np.float64)  
# sr1m = np.zeros(nmarker,dtype=np.float64)  
# sr2m = np.zeros(nmarker,dtype=np.float64)  
# sr3m = np.zeros(nmarker,dtype=np.float64)  
# sr4m = np.zeros(nmarker,dtype=np.float64)  
# sr5m = np.zeros(nmarker,dtype=np.float64)  
    
# exx0mT=0. ; exx1mT=0. ; exx2mT=0. ; exx3mT=0. ; exx4mT=0. ; exx5mT=0.
# eyy0mT=0. ; eyy1mT=0. ; eyy2mT=0. ; eyy3mT=0. ; eyy4mT=0. ; eyy5mT=0.
# exy0mT=0. ; exy1mT=0. ; exy2mT=0. ; exy3mT=0. ; exy4mT=0. ; exy5mT=0.
# sr0mT=0.  ; sr1mT=0.  ; sr2mT=0.  ; sr3mT=0.  ; sr4mT=0.  ; sr5mT=0.   

# for i in range(0,nmarker):
#     xm[i]=random.random()
#     ym[i]=random.random()
#     # localise marker
#     ielx=int(xm[i]/Lx*nelx)
#     iely=int(ym[i]/Ly*nely)
#     iel=nelx*iely+ielx
#     # find reduced coordinates
#     xmin=x[icon[0,iel]]
#     xmax=x[icon[2,iel]]
#     rm=((xm[i]-xmin)/(xmax-xmin)-0.5)*2
#     ymin=y[icon[0,iel]]
#     ymax=y[icon[2,iel]]
#     sm=((ym[i]-ymin)/(ymax-ymin)-0.5)*2
#     # evaluate shape fcts at marker
#     N[0:m]=NNV(rm,sm)
#     dNdr[0:m]=dNNVdr(rm,sm)
#     dNds[0:m]=dNNVds(rm,sm)
#     jcb=np.zeros((2,2),dtype=float)
#     for k in range(0,m):
#         jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
#         jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
#         jcb[1,0]+=dNds[k]*x[icon[k,iel]]
#         jcb[1,1]+=dNds[k]*y[icon[k,iel]]
#     jcob=np.linalg.det(jcb)
#     jcbi=np.linalg.inv(jcb)
#     for k in range(0, m):
#         dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
#         dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

#     exx0m[i]=exx[iel]
#     exx1m[i]=N[:].dot(exx1[icon[:,iel]])     
#     exx2m[i]=N[:].dot(exx2[icon[:,iel]])     
#     exx3m[i]=N[:].dot(exx3[icon[:,iel]])     
#     exx4m[i]=N[:].dot(exx4[icon[:,iel]])     
#     exx5m[i]=dNdx[:].dot(u[icon[:,iel]])

#     eyy0m[i]=eyy[iel]
#     eyy1m[i]=N[:].dot(eyy1[icon[:,iel]])     
#     eyy2m[i]=N[:].dot(eyy2[icon[:,iel]])     
#     eyy3m[i]=N[:].dot(eyy3[icon[:,iel]])     
#     eyy4m[i]=N[:].dot(eyy4[icon[:,iel]])     
#     eyy5m[i]=dNdy[:].dot(v[icon[:,iel]])

#     exy0m[i]=exy[iel]
#     exy1m[i]=N[:].dot(exy1[icon[:,iel]])     
#     exy2m[i]=N[:].dot(exy2[icon[:,iel]])     
#     exy3m[i]=N[:].dot(exy3[icon[:,iel]])     
#     exy4m[i]=N[:].dot(exy4[icon[:,iel]])     
#     exy5m[i]=dNdx[:].dot(v[icon[:,iel]])*0.5\
#             +dNdy[:].dot(u[icon[:,iel]])*0.5

#     exx0mT+=abs(exx0m[i]-exxth(xm[i],ym[i]))**2
#     exx1mT+=abs(exx1m[i]-exxth(xm[i],ym[i]))**2
#     exx2mT+=abs(exx2m[i]-exxth(xm[i],ym[i]))**2
#     exx3mT+=abs(exx3m[i]-exxth(xm[i],ym[i]))**2
#     exx4mT+=abs(exx4m[i]-exxth(xm[i],ym[i]))**2
#     exx5mT+=abs(exx5m[i]-exxth(xm[i],ym[i]))**2

#     eyy0mT+=abs(eyy0m[i]-eyyth(xm[i],ym[i]))**2
#     eyy1mT+=abs(eyy1m[i]-eyyth(xm[i],ym[i]))**2
#     eyy2mT+=abs(eyy2m[i]-eyyth(xm[i],ym[i]))**2
#     eyy3mT+=abs(eyy3m[i]-eyyth(xm[i],ym[i]))**2
#     eyy4mT+=abs(eyy4m[i]-eyyth(xm[i],ym[i]))**2
#     eyy5mT+=abs(eyy5m[i]-eyyth(xm[i],ym[i]))**2

#     exy0mT+=abs(exy0m[i]-exyth(xm[i],ym[i]))**2
#     exy1mT+=abs(exy1m[i]-exyth(xm[i],ym[i]))**2
#     exy2mT+=abs(exy2m[i]-exyth(xm[i],ym[i]))**2
#     exy3mT+=abs(exy3m[i]-exyth(xm[i],ym[i]))**2
#     exy4mT+=abs(exy4m[i]-exyth(xm[i],ym[i]))**2
#     exy5mT+=abs(exy5m[i]-exyth(xm[i],ym[i]))**2

#     sr0m[i]=np.sqrt(0.5*exx0m[i]**2 + 0.5*eyy0m[i]**2 + exy0m[i]**2 )
#     sr1m[i]=np.sqrt(0.5*exx1m[i]**2 + 0.5*eyy1m[i]**2 + exy1m[i]**2 )
#     sr2m[i]=np.sqrt(0.5*exx2m[i]**2 + 0.5*eyy2m[i]**2 + exy2m[i]**2 )
#     sr3m[i]=np.sqrt(0.5*exx3m[i]**2 + 0.5*eyy3m[i]**2 + exy3m[i]**2 )
#     sr4m[i]=np.sqrt(0.5*exx4m[i]**2 + 0.5*eyy4m[i]**2 + exy4m[i]**2 )
#     sr5m[i]=np.sqrt(0.5*exx5m[i]**2 + 0.5*eyy5m[i]**2 + exy5m[i]**2 )

#     sr0mT+=abs(sr0m[i]-srth(xm[i],ym[i]))**2
#     sr1mT+=abs(sr1m[i]-srth(xm[i],ym[i]))**2
#     sr2mT+=abs(sr2m[i]-srth(xm[i],ym[i]))**2
#     sr3mT+=abs(sr3m[i]-srth(xm[i],ym[i]))**2
#     sr4mT+=abs(sr4m[i]-srth(xm[i],ym[i]))**2
#     sr5mT+=abs(sr5m[i]-srth(xm[i],ym[i]))**2

# exx0mT/=nmarker ; exx0mT=np.sqrt(exx0mT)
# exx1mT/=nmarker ; exx1mT=np.sqrt(exx1mT)
# exx2mT/=nmarker ; exx2mT=np.sqrt(exx2mT)
# exx3mT/=nmarker ; exx3mT=np.sqrt(exx3mT)
# exx4mT/=nmarker ; exx4mT=np.sqrt(exx4mT)
# exx5mT/=nmarker ; exx5mT=np.sqrt(exx5mT)

# eyy0mT/=nmarker ; eyy0mT=np.sqrt(eyy0mT)
# eyy1mT/=nmarker ; eyy1mT=np.sqrt(eyy1mT)
# eyy2mT/=nmarker ; eyy2mT=np.sqrt(eyy2mT)
# eyy3mT/=nmarker ; eyy3mT=np.sqrt(eyy3mT)
# eyy4mT/=nmarker ; eyy4mT=np.sqrt(eyy4mT)
# eyy5mT/=nmarker ; eyy5mT=np.sqrt(eyy5mT)

# exy0mT/=nmarker ; exy0mT=np.sqrt(exy0mT)
# exy1mT/=nmarker ; exy1mT=np.sqrt(exy1mT)
# exy2mT/=nmarker ; exy2mT=np.sqrt(exy2mT)
# exy3mT/=nmarker ; exy3mT=np.sqrt(exy3mT)
# exy4mT/=nmarker ; exy4mT=np.sqrt(exy4mT)
# exy5mT/=nmarker ; exy5mT=np.sqrt(exy5mT)

# sr0mT/=nmarker ; sr0mT=np.sqrt(sr0mT)
# sr1mT/=nmarker ; sr1mT=np.sqrt(sr1mT)
# sr2mT/=nmarker ; sr2mT=np.sqrt(sr2mT)
# sr3mT/=nmarker ; sr3mT=np.sqrt(sr3mT)
# sr4mT/=nmarker ; sr4mT=np.sqrt(sr4mT)
# sr5mT/=nmarker ; sr5mT=np.sqrt(sr5mT)

# print ('nel ',nel,'avrg exx on markers ',exx0mT,exx1mT,exx2mT,exx3mT,exx4mT,exx5mT)
# print ('nel ',nel,'avrg eyy on markers ',eyy0mT,eyy1mT,eyy2mT,eyy3mT,eyy4mT,eyy5mT)
# print ('nel ',nel,'avrg exy on markers ',exy0mT,exy1mT,exy2mT,exy3mT,exy4mT,exy5mT)
# print ('nel ',nel,'avrg sr  on markers ',sr0mT,sr1mT,sr2mT,sr3mT,sr4mT,sr5mT)

# print("marker errors: %.3f s" % (time.time() - start))

# if visu==1:

#    filename = 'markers.vtu'
#    vtufile=open(filename,"w")
#    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
#    vtufile.write("<UnstructuredGrid> \n")
#    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))

#    vtufile.write("<PointData Scalars='scalars'>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx0' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx0m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx1m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx2m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx3m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx4' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx4m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx5' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exx5m[i])
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx0 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx0m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx1m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx2m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx3m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx4 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx4m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx5 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exx5m[i]-exxth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy0' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy0m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy1m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy2m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy3m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy4' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy4m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy5' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % exy5m[i])
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy0 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy0m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy1m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy2m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy3m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy4 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy4m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy5 (err)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % (exy5m[i]-exyth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr0' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr0m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr1' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr1m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr2' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr2m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr3' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr3m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr4' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr4m[i])
#    vtufile.write("</DataArray>\n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr5' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" % sr5m[i])
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sr5 (analytical)' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%15e \n" %  (srth(xm[i],ym[i])))
#    vtufile.write("</DataArray>\n")


#    vtufile.write("</PointData>\n")

#    vtufile.write("<Points> \n")
#    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
#    for i in range(0,nmarker):
#        vtufile.write("%10e %10e %10e \n" %(xm[i],ym[i],0.))
#    vtufile.write("</DataArray>\n")
#    vtufile.write("</Points> \n")

#    vtufile.write("<Cells>\n")

#    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%d " % i)
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
#    for i in range(0,nmarker):
#        vtufile.write("%d " % (i+1))
#    vtufile.write("</DataArray>\n")

#    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
#    for i in range(0,nmarker):
#        vtufile.write("%d " % 1)
#    vtufile.write("</DataArray>\n")

#    vtufile.write("</Cells>\n")

#    vtufile.write("</Piece>\n")
#    vtufile.write("</UnstructuredGrid>\n")
#    vtufile.write("</VTKFile>\n")
#    vtufile.close()


# #####################################################################

# nstep=1#35000
# dt=0.01
# xp = np.zeros(nstep,dtype=np.float64) 
# yp = np.zeros(nstep,dtype=np.float64) 

# xp[0]=0.8123
# yp[0]=0.8123

# strain0 = np.zeros(nstep,dtype=np.float64) 
# strain1 = np.zeros(nstep,dtype=np.float64) 
# strain2 = np.zeros(nstep,dtype=np.float64) 
# strain3 = np.zeros(nstep,dtype=np.float64) 
# strain4 = np.zeros(nstep,dtype=np.float64) 
# strain5 = np.zeros(nstep,dtype=np.float64) 

# for istep in range(0,nstep-1):
#     ielx=int(xp[istep]/Lx*nelx)
#     iely=int(yp[istep]/Ly*nely)
#     iel=nelx*iely+ielx
#     xmin=x[icon[0,iel]]
#     xmax=x[icon[2,iel]]
#     rm=((xm[i]-xmin)/(xmax-xmin)-0.5)*2
#     ymin=y[icon[0,iel]]
#     ymax=y[icon[2,iel]]
#     sm=((ym[i]-ymin)/(ymax-ymin)-0.5)*2
#     N[0:m]=NNV(rm,sm)
#     dNdr[0:m]=dNNVdr(rm,sm)
#     dNds[0:m]=dNNVds(rm,sm)
#     jcb=np.zeros((2,2),dtype=np.float64)
#     for k in range(0,m):
#         jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
#         jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
#         jcb[1,0]+=dNds[k]*x[icon[k,iel]]
#         jcb[1,1]+=dNds[k]*y[icon[k,iel]]
#     jcob=np.linalg.det(jcb)
#     jcbi=np.linalg.inv(jcb)
#     for k in range(0, m):
#         dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
#         dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

#     strain0[istep+1]=strain0[istep]+abs(exy[iel])
#     strain1[istep+1]=strain1[istep]+abs(N[:].dot(exy1[icon[:,iel]]))
#     strain2[istep+1]=strain2[istep]+abs(N[:].dot(exy2[icon[:,iel]]))
#     strain3[istep+1]=strain3[istep]+abs(N[:].dot(exy3[icon[:,iel]]))
#     strain4[istep+1]=strain4[istep]+abs(N[:].dot(exy4[icon[:,iel]]))
#     strain5[istep+1]=strain5[istep]+abs(dNdx[:].dot(u[icon[:,iel]]))

#     up=uth(xp[istep],yp[istep])
#     vp=vth(xp[istep],yp[istep])
#     #print (up,vp) 
#     xp[istep+1]=xp[istep]+up*dt    
#     yp[istep+1]=yp[istep]+vp*dt    


# np.savetxt('particle.ascii',np.array([xp,yp]).T,header='# x,y')
# np.savetxt('strains.ascii',np.array([strain0,strain1,strain2,strain3,strain4,strain5]).T,header='# x,y')


print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


