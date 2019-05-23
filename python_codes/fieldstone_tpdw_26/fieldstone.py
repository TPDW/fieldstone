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
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
print("variable declaration")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 96
   nely = 96
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

viscosity=1.  # dynamic viscosity \mu

NfemV=nnp*ndofV                 # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP                # total number of dofs

eps=1.e-10
sqrt3=np.sqrt(3.)

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

random_grid=True

pnormalise=True

sparse=True

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
              dx=2.*(random.random()-0.5)*hx/25
              dy=2.*(random.random()-0.5)*hy/25
           else:
              dx=0
              dy=0
           x[counter]=i*hx/2+dx
           y[counter]=j*hy/2+dy
           counter += 1
else:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*hx/2
           y[counter]=j*hy/2
           counter += 1

np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int16)
iconP=np.zeros((mP,nel),dtype=np.int16)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1


counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
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
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

if sparse:
  A_sparse = lil_matrix((Nfem,Nfem))
else:
  K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
  G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNP= np.zeros(mP*ndofP,dtype=np.float64)   

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NV[0:9]=NNV(rq,sq)
            dNVdr[0:9]=dNNVdr(rq,sq)
            dNVds[0:9]=dNNVds(rq,sq)
            NP[0:4]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
                jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
                jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
                jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.     ],
                                         [0.      ,dNVdy[i]],
                                         [dNVdy[i],dNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NV[i]*jcob*weightq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NP[i]
                N_mat[1,i]=NP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNP[:]+=NP[:]*jcob*weightq

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    #K_mat[m1,m2]+=K_el[ikk,jkk]
                    if sparse:
                      A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                      K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                #G_mat[m1,m2]+=G_el[ikk,jkk]
                if sparse:
                  A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                  A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:  
                  G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNP[k2]

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if sparse:
    rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b

else:   
  if pnormalise:
     a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
     rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
     a_mat[0:NfemV,0:NfemV]=K_mat
     a_mat[0:NfemV,NfemV:Nfem]=G_mat
     a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
     a_mat[Nfem,NfemV:Nfem]=constr
     a_mat[NfemV:Nfem,Nfem]=constr
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
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))



######################################################################
# compute elemental strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    NV[0:mV]=NNV(rq,sq)
    dNVdr[0:mV]=dNNVdr(rq,sq)
    dNVds[0:mV]=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, mV):
        jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
        jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
        jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
        jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, mV):
        dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
        dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

    for k in range(0, mV):
        xc[iel] += NV[k]*x[iconV[k,iel]]
        yc[iel] += NV[k]*y[iconV[k,iel]]
        exx[iel] += dNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
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

  for k in range(mV):

    exx1[iconV[k,iel]]+=exx[iel]
    eyy1[iconV[k,iel]]+=eyy[iel]
    exy1[iconV[k,iel]]+=exy[iel]

    count[iconV[k,iel]]+=1



exx1/=count
eyy1/=count
exy1/=count

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
  for node in range(9):
    #print(node)
    if node==0:
      #lower left
      rq=-1
      sq=-1
    if node==1:
      #lower right
      rq=+1
      sq=-1
    if node==2:
      #upper right
      rq=+1
      sq=+1
    if node==3:
      #upper left
      rq=-1
      sq=+1
    if node==4:
      #lower centre
      rq=0
      sq=-1
    if node==5:
      #right centre
      rq=+1
      sq=0
    if node==6:
      #upper centre
      rq=0
      sq=+1
    if node==7:
      #left centre
      rq=-1
      sq=0
    if node==8:
      #centre
      rq=0
      sq=0


    dNVdr[0:mV]=dNNVdr(rq,sq)
    dNVds[0:mV]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
        jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
        jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
        jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
    #     print(x[iconV[k,iel]],y[iconV[k,iel]])
    #     print(dNVds[k],dNVdr[k])
    #     print(" ")
    # print(jcb)
    #stop
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,mV):
        dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
        dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
    exx_rs=dNVdx[0:mV].dot(u[iconV[0:mV,iel]])
    eyy_rs=dNVdy[0:mV].dot(v[iconV[0:mV,iel]])
    exy_rs=dNVdx[0:mV].dot(v[iconV[0:mV,iel]])*0.5\
          +dNVdy[0:mV].dot(u[iconV[0:mV,iel]])*0.5
    exx2[iconV[node,iel]]+=exx_rs
    eyy2[iconV[node,iel]]+=eyy_rs
    exy2[iconV[node,iel]]+=exy_rs
    count[iconV[node,iel]]+=1




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
count=np.zeros(nnp,dtype=np.float64)  

mean_condition = 0
condition_count= 0


counter = 0
for j in range(0,nely):
    #print(j)
    for i in range(0,nelx):
        iel0=j*nely+i
        if i<nelx-1 and j<nely-1:
           iel0=iel0
           iel1=iel0+1
           iel2=iel0+nelx+1
           iel3=iel0+nelx
           #print(iel0,iel1,iel2,iel3)

           AA = np.zeros((9,9),dtype=np.float64) 
           BBxx = np.zeros(9,dtype=np.float64) 
           BByy = np.zeros(9,dtype=np.float64) 
           BBxy = np.zeros(9,dtype=np.float64) 


           for l in range(16):
              #I need a bit of code here to get x_i and y_i

              xi=0
              yi=0

              if l in [0,1,2,3]:
                element=iel0
              if l in [4,5,6,7]:
                element=iel1
              if l in [8,9,10,11]:
                element=iel2
              if l in [12,13,14,15]:
                element=iel3


              sconv=1/sqrt3

              if l%4 == 0:
                rq=-sconv
                sq=-sconv
              if l%4 == 1:
                rq=+sconv
                sq=-sconv
              if l%4 == 2:
                rq=+sconv
                sq=+sconv
              if l%4 == 3:
                rq=-sconv
                sq=+sconv


              NV[0:9]=NNV(rq,sq)
              dNVdr[0:9]=dNNVdr(rq,sq)
              dNVds[0:9]=dNNVds(rq,sq)

              # calculate jacobian matrix
              jcb=np.zeros((2,2),dtype=np.float64)
              for k in range(0,mV):
                  jcb[0,0] += dNVdr[k]*x[iconV[k,element]]
                  jcb[0,1] += dNVdr[k]*y[iconV[k,element]]
                  jcb[1,0] += dNVds[k]*x[iconV[k,element]]
                  jcb[1,1] += dNVds[k]*y[iconV[k,element]]
              jcob = np.linalg.det(jcb)
              jcbi = np.linalg.inv(jcb)

              # compute dNdx & dNdy
              xi=0.0
              yi=0.0
              exxi=0.0
              exyi=0.0
              eyyi=0.0
              for k in range(0,mV):
                  xi+=NV[k]*x[iconV[k,element]]
                  yi+=NV[k]*y[iconV[k,element]]
                  dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                  dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

                  exxi += dNVdx[k]*u[iconV[k,element]]
                  eyyi += dNVdy[k]*v[iconV[k,element]]
                  exyi += (dNVdx[k]*v[iconV[k,element]]+dNVdy[k]*u[iconV[k,element]])*0.5




              #Now I specify the entire 9x9 matrix by hand
              #There's probably a nicer way of doing this

              AA[0,0]+=1
              AA[1,0]+=xi
              AA[2,0]+=yi
              AA[3,0]+=xi*yi
              AA[4,0]+=xi**2
              AA[5,0]+=yi**2
              AA[6,0]+=xi**2*yi
              AA[7,0]+=xi*yi**2
              AA[8,0]+=xi**2*yi**2

              AA[0,1]+=xi*1
              AA[1,1]+=xi*xi
              AA[2,1]+=xi*yi
              AA[3,1]+=xi*xi*yi
              AA[4,1]+=xi*xi**2
              AA[5,1]+=xi*yi**2
              AA[6,1]+=xi*xi**2*yi
              AA[7,1]+=xi*xi*yi**2
              AA[8,1]+=xi*xi**2*yi**2

              AA[0,2]+=yi*1
              AA[1,2]+=yi*xi
              AA[2,2]+=yi*yi
              AA[3,2]+=yi*xi*yi
              AA[4,2]+=yi*xi**2
              AA[5,2]+=yi*yi**2
              AA[6,2]+=yi*xi**2*yi
              AA[7,2]+=yi*xi*yi**2
              AA[8,2]+=yi*xi**2*yi**2

              AA[0,3]+=xi*yi*1
              AA[1,3]+=xi*yi*xi
              AA[2,3]+=xi*yi*yi
              AA[3,3]+=xi*yi*xi*yi
              AA[4,3]+=xi*yi*xi**2
              AA[5,3]+=xi*yi*yi**2
              AA[6,3]+=xi*yi*xi**2*yi
              AA[7,3]+=xi*yi*xi*yi**2
              AA[8,3]+=xi*yi*xi**2*yi**2

              AA[0,4]+=xi**2*1
              AA[1,4]+=xi**2*xi
              AA[2,4]+=xi**2*yi
              AA[3,4]+=xi**2*xi*yi
              AA[4,4]+=xi**2*xi**2
              AA[5,4]+=xi**2*yi**2
              AA[6,4]+=xi**2*xi**2*yi
              AA[7,4]+=xi**2*xi*yi**2
              AA[8,4]+=xi**2*xi**2*yi**2

              AA[0,5]+=yi**2*1
              AA[1,5]+=yi**2*xi
              AA[2,5]+=yi**2*yi
              AA[3,5]+=yi**2*xi*yi
              AA[4,5]+=yi**2*xi**2
              AA[5,5]+=yi**2*yi**2
              AA[6,5]+=yi**2*xi**2*yi
              AA[7,5]+=yi**2*xi*yi**2
              AA[8,5]+=yi**2*xi**2*yi**2

              AA[0,6]+=xi**2*yi*1
              AA[1,6]+=xi**2*yi*xi
              AA[2,6]+=xi**2*yi*yi
              AA[3,6]+=xi**2*yi*xi*yi
              AA[4,6]+=xi**2*yi*xi**2
              AA[5,6]+=xi**2*yi*yi**2
              AA[6,6]+=xi**2*yi*xi**2*yi
              AA[7,6]+=xi**2*yi*xi*yi**2
              AA[8,6]+=xi**2*yi*xi**2*yi**2

              AA[0,7]+=xi*yi**2*1
              AA[1,7]+=xi*yi**2*xi
              AA[2,7]+=xi*yi**2*yi
              AA[3,7]+=xi*yi**2*xi*yi
              AA[4,7]+=xi*yi**2*xi**2
              AA[5,7]+=xi*yi**2*yi**2
              AA[6,7]+=xi*yi**2*xi**2*yi
              AA[7,7]+=xi*yi**2*xi*yi**2
              AA[8,7]+=xi*yi**2*xi**2*yi**2

              AA[0,8]+=xi**2*yi**2*1
              AA[1,8]+=xi**2*yi**2*xi
              AA[2,8]+=xi**2*yi**2*yi
              AA[3,8]+=xi**2*yi**2*xi*yi
              AA[4,8]+=xi**2*yi**2*xi**2
              AA[5,8]+=xi**2*yi**2*yi**2
              AA[6,8]+=xi**2*yi**2*xi**2*yi
              AA[7,8]+=xi**2*yi**2*xi*yi**2
              AA[8,8]+=xi**2*yi**2*xi**2*yi**2





              BBxx[0]+=exxi*1 
              BBxx[1]+=exxi*xi 
              BBxx[2]+=exxi*yi
              BBxx[3]+=exxi*xi*yi
              BBxx[4]+=exxi*xi**2 
              BBxx[5]+=exxi*yi**2 
              BBxx[6]+=exxi*xi**2*yi
              BBxx[7]+=exxi*xi*yi**2
              BBxx[8]+=exxi*xi**2*yi**2 

              BByy[0]+=eyyi*1 
              BByy[1]+=eyyi*xi 
              BByy[2]+=eyyi*yi
              BByy[3]+=eyyi*xi*yi
              BByy[4]+=eyyi*xi**2 
              BByy[5]+=eyyi*yi**2 
              BByy[6]+=eyyi*xi**2*yi
              BByy[7]+=eyyi*xi*yi**2
              BByy[8]+=eyyi*xi**2*yi**2 

              BBxy[0]+=exyi*1 
              BBxy[1]+=exyi*xi 
              BBxy[2]+=exyi*yi
              BBxy[3]+=exyi*xi*yi
              BBxy[4]+=exyi*xi**2 
              BBxy[5]+=exyi*yi**2 
              BBxy[6]+=exyi*xi**2*yi
              BBxy[7]+=exyi*xi*yi**2
              BBxy[8]+=exyi*xi**2*yi**2 

           # print(np.linalg.cond(AA))
           # print("\n")

           mean_condition += np.linalg.cond(AA)
           condition_count +=1

           solxx=sps.linalg.spsolve(sps.csr_matrix(AA),BBxx)


           solyy=sps.linalg.spsolve(sps.csr_matrix(AA),BByy)


           solxy=sps.linalg.spsolve(sps.csr_matrix(AA),BBxy)



           for node in range(25):#loop over every node in the patch
              assemble=False

              if node==0:
                ip=iconV[0,iel0]
                if on_bd[ip,2] and on_bd[ip,0]:
                  assemble=True

              if node==1:
                ip=iconV[4,iel0]
                if on_bd[ip,2]:
                  assemble=True

              if node==2:
                ip=iconV[1,iel0]
                if on_bd[ip,2]:
                  assemble=True

              if node==3:
                ip=iconV[4,iel1]
                if on_bd[ip,2]:
                  assemble=True

              if node==4:
                ip=iconV[1,iel1]
                if on_bd[ip,2] and on_bd[ip,1]:
                  assemble=True

              if node==5:
                ip=iconV[7,iel0]
                if on_bd[ip,0]:
                  assemble=True

              if node==6:
                ip=iconV[8,iel0]
                assemble=True

              if node==7:
                ip=iconV[5,iel0]
                assemble=True

              if node==8:
                ip=iconV[8,iel1]
                assemble=True

              if node==9:
                ip=iconV[5,iel1]
                if on_bd[ip,1]:
                  assemble=True

              if node==10:
                ip=iconV[3,iel0]
                if on_bd[ip,0]:
                  assemble=True

              if node==11:
                ip=iconV[6,iel0]
                assemble=True

              if node==12:
                ip=iconV[2,iel0]
                assemble=True

              if node==13:
                ip=iconV[6,iel1]
                assemble=True

              if node==14:
                ip=iconV[2,iel1]
                if on_bd[ip,1]:
                  assemble=True

              if node==15:
                ip=iconV[7,iel3]
                if on_bd[ip,0]:
                  assemble=True

              if node==16:
                ip=iconV[8,iel3]
                assemble=True

              if node==17:
                ip=iconV[5,iel3]
                assemble=True

              if node==18:
                ip=iconV[8,iel2]
                assemble=True

              if node==19:
                ip=iconV[5,iel2]
                if on_bd[ip,1]:
                  assemble=True

              if node==20:
                ip=iconV[3,iel3]
                if on_bd[ip,0] and on_bd[ip,3]:
                  assemble=True

              if node==21:
                ip=iconV[6,iel3]
                if on_bd[ip,3]:
                  assemble=True

              if node==22:
                ip=iconV[2,iel3]
                if on_bd[ip,3]:
                  assemble=True

              if node==23:
                ip=iconV[6,iel2]
                if on_bd[ip,3]:
                  assemble=True

              if node==24:
                ip=iconV[2,iel2]
                if on_bd[ip,3] and on_bd[ip,1]:
                  assemble=True

              assemble=True
              if assemble:
               exx3[ip]+=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]+solxx[4]*x[ip]**2 \
               +solxx[5]*y[ip]**2+solxx[6]*x[ip]**2*y[ip]+solxx[7]*x[ip]*y[ip]**2+solxx[8]*x[ip]**2*y[ip]**2
               
               eyy3[ip]+=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]+solyy[4]*x[ip]**2 \
               +solyy[5]*y[ip]**2+solyy[6]*x[ip]**2*y[ip]+solyy[7]*x[ip]*y[ip]**2+solyy[8]*x[ip]**2*y[ip]**2
               
               exy3[ip]+=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]+solxy[4]*x[ip]**2 \
               +solxy[5]*y[ip]**2+solxy[6]*x[ip]**2*y[ip]+solxy[7]*x[ip]*y[ip]**2+solxy[8]*x[ip]**2*y[ip]**2

               count[ip] +=1
           

           counter+=1

exx3/=count
eyy3/=count
exy3/=count
mean_condition /=condition_count

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

# A_mat=np.zeros((nnp,nnp),dtype=np.float64) # Q1 mass matrix
# rhs_xx=np.zeros(nnp,dtype=np.float64)      # rhs
# rhs_yy=np.zeros(nnp,dtype=np.float64)      # rhs
# rhs_xy=np.zeros(nnp,dtype=np.float64)      # rhs
# rhs_p=np.zeros(nnp,dtype=np.float64)       # rhs

 
# for iel in range(0, nel):

#     # set arrays to 0 every loop
#     fp_el =np.zeros(m,dtype=np.float64)
#     fxx_el =np.zeros(m,dtype=np.float64)
#     fyy_el =np.zeros(m,dtype=np.float64)
#     fxy_el =np.zeros(m,dtype=np.float64)
#     M_el =np.zeros((m,m),dtype=np.float64)

#     # integrate viscous term at 4 quadrature points
#     for iq in [-1, 1]:
#         for jq in [-1, 1]:

#             # position & weight of quad. point
#             rq=iq/sqrt3
#             sq=jq/sqrt3
#             wq=1.*1.

#             # calculate shape functions
#             N[0:m]=NNV(rq,sq)
#             dNVdr[0:m]=dNNVdr(rq,sq)
#             dNVds[0:m]=dNNVds(rq,sq)

#             # calculate jacobian matrix
#             jcb = np.zeros((2,2),dtype=np.float64)
#             for k in range(0,m):
#                 jcb[0, 0] += dNVdr[k]*x[icon[k,iel]]
#                 jcb[0, 1] += dNVdr[k]*y[icon[k,iel]]
#                 jcb[1, 0] += dNVds[k]*x[icon[k,iel]]
#                 jcb[1, 1] += dNVds[k]*y[icon[k,iel]]
#             jcob = np.linalg.det(jcb)
#             jcbi = np.linalg.inv(jcb)

#             # compute dNdx & dNdy
#             exxq=0.0
#             eyyq=0.0
#             exyq=0.0
#             for k in range(0, m):
#                 dNdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
#                 dNdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
#                 exxq += dNdx[k]*u[icon[k,iel]]
#                 eyyq += dNdy[k]*v[icon[k,iel]]
#                 exyq += (dNdx[k]*v[icon[k,iel]]+dNdy[k]*u[icon[k,iel]])*0.5

#             for i in range(0,m):
#                 for j in range(0,m):
#                     M_el[i,j]+=N[i]*N[j]*wq*jcob
#                 if not reduced:
#                    fxx_el[i]+=N[i]*exxq*wq*jcob
#                    fyy_el[i]+=N[i]*eyyq*wq*jcob
#                    fxy_el[i]+=N[i]*exyq*wq*jcob
#                    fp_el[i]+=N[i]*p[iel]*wq*jcob

#     if reduced:
#        rq=0.
#        sq=0.
#        wq=2.*2.
#        N[0:m]=NNV(rq,sq)
#        dNVdr[0:m]=dNNVdr(rq,sq)
#        dNVds[0:m]=dNNVds(rq,sq)
#        jcb = np.zeros((2,2),dtype=np.float64)
#        for k in range(0,m):
#            jcb[0,0]+=dNVdr[k]*x[icon[k,iel]]
#            jcb[0,1]+=dNVdr[k]*y[icon[k,iel]]
#            jcb[1,0]+=dNVds[k]*x[icon[k,iel]]
#            jcb[1,1]+=dNVds[k]*y[icon[k,iel]]
#        jcob=np.linalg.det(jcb)
#        jcbi=np.linalg.inv(jcb)
#        exxq=0.0
#        eyyq=0.0
#        exyq=0.0
#        for k in range(0, m):
#            dNdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
#            dNdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
#            exxq += dNdx[k]*u[icon[k,iel]]
#            eyyq += dNdy[k]*v[icon[k,iel]]
#            exyq += (dNdx[k]*v[icon[k,iel]]+dNdy[k]*u[icon[k,iel]])*0.5
#        for i in range(0,m):
#            fxx_el[i]+=N[i]*exxq*wq*jcob
#            fyy_el[i]+=N[i]*eyyq*wq*jcob
#            fxy_el[i]+=N[i]*exyq*wq*jcob
#            fp_el[i]+=N[i]*p[iel]*wq*jcob

#     # assemble matrix a_mat and right hand side rhs
#     for k1 in range(0,m):
#         ik=icon[k1,iel]
#         for k2 in range(0,m):
#             jk=icon[k2,iel]
#             A_mat[ik,jk]+=M_el[k1,k2]
#         rhs_xx[ik]+=fxx_el[k1]
#         rhs_yy[ik]+=fyy_el[k1]
#         rhs_xy[ik]+=fxy_el[k1]
#         rhs_p[ik]+=fp_el[k1]

# exx4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xx)
# eyy4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_yy)
# exy4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xy)
# p4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_p)

print("     -> exx4 (m,M) %.4f %.4f " %(np.min(exx4),np.max(exx4)))
print("     -> eyy4 (m,M) %.4f %.4f " %(np.min(eyy4),np.max(eyy4)))
print("     -> exy4 (m,M) %.4f %.4f " %(np.min(exy4),np.max(exy4)))

np.savetxt('srn_4.ascii',np.array([x,y,exx4,eyy4,exy4]).T,header='# x,y,exx4,eyy4,exy4')
np.savetxt('p_4.ascii',np.array([x,y,p4]).T,header='# x,y,p')

######################################################################
# compute error
######################################################################
start = time.time()

errv=0.
errexx0=0. ; errexx1=0. ; errexx2=0. ; errexx3=0. ; errexx4=0. ; errexx5=0.
erreyy0=0. ; erreyy1=0. ; erreyy2=0. ; erreyy3=0. ; erreyy4=0. ; erreyy5=0.
errexy0=0. ; errexy1=0. ; errexy2=0. ; errexy3=0. ; errexy4=0. ; errexy5=0.

degree=3
gleg_points,gleg_weights=np.polynomial.legendre.leggauss(degree)

for iel in range (0,nel):
    for iq in range(degree):
        for jq in range(degree):
            rq=gleg_points[iq]
            sq=gleg_points[jq]
            weightq=gleg_weights[iq]*gleg_weights[jq]
            NV[0:mV]=NNV(rq,sq)
            dNVdr[0:mV]=dNNVdr(rq,sq)
            dNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
                jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
                jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
                jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0, mV):
                dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            exx1q=0. ; exx2q=0. ; exx3q=0. ; exx4q=0. ; exx5q=0.
            eyy1q=0. ; eyy2q=0. ; eyy3q=0. ; eyy4q=0. ; eyy5q=0.
            exy1q=0. ; exy2q=0. ; exy3q=0. ; exy4q=0. ; exy5q=0.
            for k in range(0,mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                uq+=NV[k]*u[iconV[k,iel]]
                vq+=NV[k]*v[iconV[k,iel]]
                exx1q+=NV[k]*exx1[iconV[k,iel]]
                exx2q+=NV[k]*exx2[iconV[k,iel]]
                exx3q+=NV[k]*exx3[iconV[k,iel]]
                exx4q+=NV[k]*exx4[iconV[k,iel]]
                exx5q+=dNVdx[k]*u[iconV[k,iel]]
                eyy1q+=NV[k]*eyy1[iconV[k,iel]]
                eyy2q+=NV[k]*eyy2[iconV[k,iel]]
                eyy3q+=NV[k]*eyy3[iconV[k,iel]]
                eyy4q+=NV[k]*eyy4[iconV[k,iel]]
                eyy5q+=dNVdy[k]*v[iconV[k,iel]]
                exy1q+=NV[k]*exy1[iconV[k,iel]]
                exy2q+=NV[k]*exy2[iconV[k,iel]]
                exy3q+=NV[k]*exy3[iconV[k,iel]]
                exy4q+=NV[k]*exy4[iconV[k,iel]]
                exy5q+=dNVdx[k]*v[iconV[k,iel]]*0.5\
                      +dNVdy[k]*u[iconV[k,iel]]*0.5
            exx0q=exx[iel]
            eyy0q=eyy[iel]
            exy0q=exy[iel]
            errv+=((uq-uth(xq,yq))**2+(vq-vth(xq,yq))**2)*weightq*jcob
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


print(type(np.sqrt))
print(type(errv))

errv=np.sqrt(errv)

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

print("     -> nel= %6d ; errv= %.8e" %(nel,errv))
print("     -> nel= %6d ; errexx0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,errexx0,errexx1,errexx2,errexx3,errexx4,errexx5))
print("     -> nel= %6d ; erreyy0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,erreyy0,erreyy1,erreyy2,erreyy3,erreyy4,erreyy5))
print("     -> nel= %6d ; errexy0,1,2,3,4,5 %.8e %.8e %.8e %.8e %.8e %.8e" %(nel,errexy0,errexy1,errexy2,errexy3,errexy4,errexy5))
print("     -> nel= %6d ; condition number %10.3E" %(nel, mean_condition))

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
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
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

#####################################################################
# using markers  
#####################################################################
# start = time.time()

# if random_grid:
#    print ("random grid is on!!")

# nmarker=1000000

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
#     dNVdr[0:m]=dNNVdr(rm,sm)
#     dNVds[0:m]=dNNVds(rm,sm)
#     jcb=np.zeros((2,2),dtype=float)
#     for k in range(0,m):
#         jcb[0,0]+=dNVdr[k]*x[icon[k,iel]]
#         jcb[0,1]+=dNVdr[k]*y[icon[k,iel]]
#         jcb[1,0]+=dNVds[k]*x[icon[k,iel]]
#         jcb[1,1]+=dNVds[k]*y[icon[k,iel]]
#     jcob=np.linalg.det(jcb)
#     jcbi=np.linalg.inv(jcb)
#     for k in range(0, m):
#         dNdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
#         dNdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

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
#     dNVdr[0:m]=dNNVdr(rm,sm)
#     dNVds[0:m]=dNNVds(rm,sm)
#     jcb=np.zeros((2,2),dtype=np.float64)
#     for k in range(0,m):
#         jcb[0,0]+=dNVdr[k]*x[icon[k,iel]]
#         jcb[0,1]+=dNVdr[k]*y[icon[k,iel]]
#         jcb[1,0]+=dNVds[k]*x[icon[k,iel]]
#         jcb[1,1]+=dNVds[k]*y[icon[k,iel]]
#     jcob=np.linalg.det(jcb)
#     jcbi=np.linalg.inv(jcb)
#     for k in range(0, m):
#         dNdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
#         dNdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

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


