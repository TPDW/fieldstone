import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt
from sympy.integrals.quadrature import gauss_lobatto

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

def NNV(rq,sq):
    N1r=(-1    +rq +9*rq**2 - 9*rq**3)/16
    N2r=(+9 -27*rq -9*rq**2 +27*rq**3)/16
    N3r=(+9 +27*rq -9*rq**2 -27*rq**3)/16
    N4r=(-1    -rq +9*rq**2 + 9*rq**3)/16
    N1t=(-1    +sq +9*sq**2 - 9*sq**3)/16
    N2t=(+9 -27*sq -9*sq**2 +27*sq**3)/16
    N3t=(+9 +27*sq -9*sq**2 -27*sq**3)/16
    N4t=(-1    -sq +9*sq**2 + 9*sq**3)/16
    NV_00= N1r*N1t 
    NV_01= N2r*N1t 
    NV_02= N3r*N1t 
    NV_03= N4r*N1t 
    NV_04= N1r*N2t 
    NV_05= N2r*N2t 
    NV_06= N3r*N2t 
    NV_07= N4r*N2t 
    NV_08= N1r*N3t 
    NV_09= N2r*N3t 
    NV_10= N3r*N3t 
    NV_11= N4r*N3t 
    NV_12= N1r*N4t 
    NV_13= N2r*N4t 
    NV_14= N3r*N4t 
    NV_15= N4r*N4t 
    return NV_00,NV_01,NV_02,NV_03,NV_04,NV_05,NV_06,NV_07,\
           NV_08,NV_09,NV_10,NV_11,NV_12,NV_13,NV_14,NV_15

def dNNVdr(rq,sq):
    dN1rdr=( +1 +18*rq -27*rq**2)/16
    dN2rdr=(-27 -18*rq +81*rq**2)/16
    dN3rdr=(+27 -18*rq -81*rq**2)/16
    dN4rdr=( -1 +18*rq +27*rq**2)/16
    N1s=(-1    +sq +9*sq**2 - 9*sq**3)/16
    N2s=(+9 -27*sq -9*sq**2 +27*sq**3)/16
    N3s=(+9 +27*sq -9*sq**2 -27*sq**3)/16
    N4s=(-1    -sq +9*sq**2 + 9*sq**3)/16
    dNVdr_00= dN1rdr* N1s 
    dNVdr_01= dN2rdr* N1s 
    dNVdr_02= dN3rdr* N1s 
    dNVdr_03= dN4rdr* N1s 
    dNVdr_04= dN1rdr* N2s 
    dNVdr_05= dN2rdr* N2s 
    dNVdr_06= dN3rdr* N2s 
    dNVdr_07= dN4rdr* N2s 
    dNVdr_08= dN1rdr* N3s 
    dNVdr_09= dN2rdr* N3s 
    dNVdr_10= dN3rdr* N3s 
    dNVdr_11= dN4rdr* N3s 
    dNVdr_12= dN1rdr* N4s 
    dNVdr_13= dN2rdr* N4s 
    dNVdr_14= dN3rdr* N4s 
    dNVdr_15= dN4rdr* N4s 
    return dNVdr_00,dNVdr_01,dNVdr_02,dNVdr_03,dNVdr_04,dNVdr_05,dNVdr_06,dNVdr_07,\
           dNVdr_08,dNVdr_09,dNVdr_10,dNVdr_11,dNVdr_12,dNVdr_13,dNVdr_14,dNVdr_15

def dNNVds(rq,sq):
    N1r=(-1    +rq +9*rq**2 - 9*rq**3)/16
    N2r=(+9 -27*rq -9*rq**2 +27*rq**3)/16
    N3r=(+9 +27*rq -9*rq**2 -27*rq**3)/16
    N4r=(-1    -rq +9*rq**2 + 9*rq**3)/16
    dN1sds=( +1 +18*sq -27*sq**2)/16
    dN2sds=(-27 -18*sq +81*sq**2)/16
    dN3sds=(+27 -18*sq -81*sq**2)/16
    dN4sds=( -1 +18*sq +27*sq**2)/16
    dNVds_00= N1r*dN1sds 
    dNVds_01= N2r*dN1sds 
    dNVds_02= N3r*dN1sds 
    dNVds_03= N4r*dN1sds 
    dNVds_04= N1r*dN2sds 
    dNVds_05= N2r*dN2sds 
    dNVds_06= N3r*dN2sds 
    dNVds_07= N4r*dN2sds 
    dNVds_08= N1r*dN3sds 
    dNVds_09= N2r*dN3sds 
    dNVds_10= N3r*dN3sds 
    dNVds_11= N4r*dN3sds 
    dNVds_12= N1r*dN4sds 
    dNVds_13= N2r*dN4sds 
    dNVds_14= N3r*dN4sds 
    dNVds_15= N4r*dN4sds
    return dNVds_00,dNVds_01,dNVds_02,dNVds_03,dNVds_04,dNVds_05,dNVds_06,dNVds_07,\
           dNVds_08,dNVds_09,dNVds_10,dNVds_11,dNVds_12,dNVds_13,dNVds_14,dNVds_15

def NNP(rq,sq):
    NP_0= 0.5*rq*(rq-1) * 0.5*sq*(sq-1)
    NP_1=     (1-rq**2) * 0.5*sq*(sq-1)
    NP_2= 0.5*rq*(rq+1) * 0.5*sq*(sq-1)
    NP_3= 0.5*rq*(rq-1) *     (1-sq**2)
    NP_4=     (1-rq**2) *     (1-sq**2)
    NP_5= 0.5*rq*(rq+1) *     (1-sq**2)
    NP_6= 0.5*rq*(rq-1) * 0.5*sq*(sq+1)
    NP_7=     (1-rq**2) * 0.5*sq*(sq+1)
    NP_8= 0.5*rq*(rq+1) * 0.5*sq*(sq+1)
    return NP_0,NP_1,NP_2,NP_3,NP_4,NP_5,NP_6,NP_7,NP_8


def sigma_xx(x,y):
  return 2*x**2*(2*x - 2)*(4*y**3 - 6*y**2 + 2*y) + 4*x*(-x + 1)**2*(4*y**3 - 6*y**2 + 2*y) - x*(-x + 1) + 1/6

def sigma_xy(x,y):
  return x**2*(-x + 1)**2*(12*y**2 - 12*y + 2) - y**2*(-y + 1)**2*(12*x**2 - 12*x + 2)

def sigma_yy(x,y):
  return -x*(-x + 1) - 2*y**2*(2*y - 2)*(4*x**3 - 6*x**2 + 2*x) - 4*y*(-y + 1)**2*(4*x**3 - 6*x**2 + 2*x) + 1/6

def sigma_yx(x,y):
  return sigma_xy(x,y)

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=16    # number of velocity nodes making up an element
mP=9     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 10
   nely = 10
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=3*nelx+1  # number of elements, x direction
nny=3*nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

viscosity=1.  # dynamic viscosity \mu

NfemV=nnp*ndofV               # number of velocity dofs
NfemP=(2*nelx+1)*(2*nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs

eps=1.e-10

qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
qw4a=(18-np.sqrt(30.))/36.
qw4b=(18+np.sqrt(30.))/36.
qcoords=[-qc4a,-qc4b,qc4b,qc4a]
qweights=[qw4a,qw4b,qw4b,qw4a]

hx=Lx/nelx
hy=Ly/nely

pnormalise=True

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnp=",nnp)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(nnp,dtype=np.float64)  # x coordinates
y=np.empty(nnp,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx/3.
        y[counter]=j*hy/3.
        counter += 1

np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int16)
iconP=np.zeros((mP,nel),dtype=np.int16)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[ 0,counter]=(i)*3+1+(j)*3*nnx+0*nnx -1
        iconV[ 1,counter]=(i)*3+2+(j)*3*nnx+0*nnx -1
        iconV[ 2,counter]=(i)*3+3+(j)*3*nnx+0*nnx -1
        iconV[ 3,counter]=(i)*3+4+(j)*3*nnx+0*nnx -1

        iconV[ 4,counter]=(i)*3+1+(j)*3*nnx+1*nnx -1
        iconV[ 5,counter]=(i)*3+2+(j)*3*nnx+1*nnx -1
        iconV[ 6,counter]=(i)*3+3+(j)*3*nnx+1*nnx -1
        iconV[ 7,counter]=(i)*3+4+(j)*3*nnx+1*nnx -1

        iconV[ 8,counter]=(i)*3+1+(j)*3*nnx+2*nnx -1
        iconV[ 9,counter]=(i)*3+2+(j)*3*nnx+2*nnx -1
        iconV[10,counter]=(i)*3+3+(j)*3*nnx+2*nnx -1
        iconV[11,counter]=(i)*3+4+(j)*3*nnx+2*nnx -1

        iconV[12,counter]=(i)*3+1+(j)*3*nnx+3*nnx -1
        iconV[13,counter]=(i)*3+2+(j)*3*nnx+3*nnx -1
        iconV[14,counter]=(i)*3+3+(j)*3*nnx+3*nnx -1
        iconV[15,counter]=(i)*3+4+(j)*3*nnx+3*nnx -1

        counter += 1

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print (iconV[0,iel],"at pos.",x[iconV[0,iel]], y[iconV[0,iel]])
#    print (iconV[1,iel],"at pos.",x[iconV[1,iel]], y[iconV[1,iel]])
#    print (iconV[2,iel],"at pos.",x[iconV[2,iel]], y[iconV[2,iel]])
#    print (iconV[3,iel],"at pos.",x[iconV[3,iel]], y[iconV[3,iel]])
#    print (iconV[4,iel],"at pos.",x[iconV[4,iel]], y[iconV[4,iel]])
#    print (iconV[5,iel],"at pos.",x[iconV[5,iel]], y[iconV[5,iel]])
#    print (iconV[6,iel],"at pos.",x[iconV[6,iel]], y[iconV[6,iel]])
#    print (iconV[7,iel],"at pos.",x[iconV[7,iel]], y[iconV[7,iel]])
#    print (iconV[8,iel],"at pos.",x[iconV[8,iel]], y[iconV[8,iel]])
#    print (iconV[9,iel],"at pos.",x[iconV[9,iel]], y[iconV[9,iel]])
#    print (iconV[10,iel],"at pos.",x[iconV[10,iel]], y[iconV[10,iel]])
#    print (iconV[11,iel],"at pos.",x[iconV[11,iel]], y[iconV[11,iel]])
#    print (iconV[12,iel],"at pos.",x[iconV[12,iel]], y[iconV[12,iel]])
#    print (iconV[13,iel],"at pos.",x[iconV[13,iel]], y[iconV[13,iel]])
#    print (iconV[14,iel],"at pos.",x[iconV[14,iel]], y[iconV[14,iel]])
#    print (iconV[15,iel],"at pos.",x[iconV[15,iel]], y[iconV[15,iel]])

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=(i)*2+1+(j)*2*(2*nelx+1) -1
        iconP[1,counter]=(i)*2+2+(j)*2*(2*nelx+1) -1
        iconP[2,counter]=(i)*2+3+(j)*2*(2*nelx+1) -1
        iconP[3,counter]=(i)*2+1+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        iconP[4,counter]=(i)*2+2+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        iconP[5,counter]=(i)*2+3+(j)*2*(2*nelx+1)+(2*nelx+1) -1
        iconP[6,counter]=(i)*2+1+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        iconP[7,counter]=(i)*2+2+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        iconP[8,counter]=(i)*2+3+(j)*2*(2*nelx+1)+(2*nelx+1)*2 -1
        counter += 1

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print (iconP[0,iel])
#    print (iconP[1,iel])
#    print (iconP[2,iel])
#    print (iconP[3,iel])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
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
print (NfemTr)

bc_nb=np.zeros(NfemV,dtype=np.int32)  # boundary condition, yes/no

counter=0
for i in range(0,NfemV):
    if (bc_fix[i]):
       # print(i)
       # print(counter)
       # print(" ")
       bc_nb[i]=counter
       counter+=1

print (np.min(bc_nb))
print (np.max(bc_nb))
np.savetxt("bc_nb.dat",bc_nb)
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
    for iq in [0,1,2,3]:
        for jq in [0,1,2,3]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NV[0:16]=NNV(rq,sq)
            dNVdr[0:16]=dNNVdr(rq,sq)
            dNVds[0:16]=dNNVds(rq,sq)
            NP[0:9]=NNP(rq,sq)

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
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
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

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

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
# compute strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    weightq=2.*2.

    NV[0:16]=NNV(rq,sq)
    dNVdr[0:16]=dNNVdr(rq,sq)
    dNVds[0:16]=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
        jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
        jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
        jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
        dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

    for k in range(0,mV):
        xc[iel] += NV[k]*x[iconV[k,iel]]
        yc[iel] += NV[k]*y[iconV[k,iel]]
        exx[iel] += dNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute error
######################################################################
start = time.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [0,1,2,3]:
        for jq in [0,1,2,3]:

            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NV[0:16]=NNV(rq,sq)
            dNVdr[0:16]=dNNVdr(rq,sq)
            dNVds[0:16]=dNNVds(rq,sq)
            NP[0:9]=NNP(rq,sq)

            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
                jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
                jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
                jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
            jcob=np.linalg.det(jcb)

            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                uq+=NV[k]*u[iconV[k,iel]]
                vq+=NV[k]*v[iconV[k,iel]]
            errv+=((uq-uth(xq,yq))**2+\
                   (vq-vth(xq,yq))**2)*weightq*jcob

            pq=0.0
            for k in range(0,mP):
                pq+=NP[k]*p[iconP[k,iel]]
            errp+=(pq-pth(xq,yq))**2*weightq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.11f ; errp= %.11f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################

q=np.zeros(nnp,dtype=np.float64)

for iel in range(0,nel):

    #node 00 ------------------------------------
    q[iconV[0,iel]]=p[iconP[0,iel]]
    #node 01 ------------------------------------
    rq=-1./3.
    sq=-1.
    NP[0:9]=NNP(rq,sq)
    q[iconV[1,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 02 ------------------------------------
    rq=+1./3.
    sq=-1.
    NP[0:9]=NNP(rq,sq)
    q[iconV[2,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 03 ------------------------------------
    q[iconV[3,iel]]=p[iconP[2,iel]]
    #node 04 ------------------------------------
    rq=-1.
    sq=-1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[4,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 05 ------------------------------------
    rq=-1./3.
    sq=-1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[5,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 06 ------------------------------------
    rq=+1./3.
    sq=-1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[6,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 07 ------------------------------------
    rq=+1.
    sq=-1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[7,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 08 ------------------------------------
    rq=-1.
    sq=+1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[8,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 09 ------------------------------------
    rq=-1./3.
    sq=+1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[9,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 10 ------------------------------------
    rq=+1./3.
    sq=+1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[10,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 11 ------------------------------------
    rq=+1.
    sq=+1./3.
    NP[0:9]=NNP(rq,sq)
    q[iconV[11,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 12 ------------------------------------
    q[iconV[12,iel]]=p[iconP[6,iel]]
    #node 13 ------------------------------------
    rq=-1./3.
    sq=+1.
    NP[0:9]=NNP(rq,sq)
    q[iconV[13,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 14 ------------------------------------
    rq=+1./3.
    sq=+1.
    NP[0:9]=NNP(rq,sq)
    q[iconV[14,iel]]=p[iconP[0:9,iel]].dot(NP[0:9])
    #node 15 ------------------------------------
    q[iconV[15,iel]]=p[iconP[8,iel]]

np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

#####################################################################
# Consistent Boundary Flux method
#####################################################################
# we wish to compute sigma_yy on the top boundary only
# this boundary counts nnx nodes, each with 1 dof for ty

M_prime = np.zeros((NfemTr,NfemTr),np.float64)
rhs_cbf = np.zeros(NfemTr,np.float64)
tx = np.zeros(nnp,np.float64)
ty = np.zeros(nnp,np.float64)


# M_prime_el=(hx/2)*np.array([\
# [16/105,957/2240,-3/70,19/840],\
# [957/2240,2619/896,-1161/2240,-327/2240],\
# [-3/70,-1161/2240,27/35,33/280],\
# [19/840,-327/2240,33/280,16/105]])


M_prime_el=np.zeros((4,4),dtype=np.float64)
use_mass_lumping=False
if use_mass_lumping:
    degree=4
    gleg_points,gleg_weights=gauss_lobatto(degree,20)
else:
    degree=14
    gleg_points,gleg_weights=np.polynomial.legendre.leggauss(degree)
for iq in range(degree):
    jq=0
    rq=gleg_points[iq]
    sq=-1
    weightq=gleg_weights[iq]
    
    NV[0:mV]=NNV(rq,sq)

    M_prime_el[0,0] += NV[0]*NV[0]*weightq
    M_prime_el[0,1] += NV[0]*NV[1]*weightq
    M_prime_el[0,2] += NV[0]*NV[2]*weightq
    M_prime_el[0,3] += NV[0]*NV[3]*weightq

    M_prime_el[1,0] += NV[1]*NV[0]*weightq
    M_prime_el[1,1] += NV[1]*NV[1]*weightq
    M_prime_el[1,2] += NV[1]*NV[2]*weightq
    M_prime_el[1,3] += NV[1]*NV[3]*weightq

    M_prime_el[2,0] += NV[2]*NV[0]*weightq
    M_prime_el[2,1] += NV[2]*NV[1]*weightq
    M_prime_el[2,2] += NV[2]*NV[2]*weightq
    M_prime_el[2,3] += NV[2]*NV[3]*weightq

    M_prime_el[3,0] += NV[3]*NV[0]*weightq
    M_prime_el[3,1] += NV[3]*NV[1]*weightq
    M_prime_el[3,2] += NV[3]*NV[2]*weightq
    M_prime_el[3,3] += NV[3]*NV[3]*weightq

M_prime_el*=0.5

for iel in range(0,nel):

    #-----------------------
    # compute Kel, Gel, f
    #-----------------------

    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    rhs_el =np.zeros((mV*ndofV),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2,3]:
        for jq in [0,1,2,3]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            # calculate shape functions
            NV[0:mV]=NNV(rq,sq)
            dNVdr[0:mV]=dNNVdr(rq,sq)
            dNVds[0:mV]=dNNVds(rq,sq)
            NP[0:mP]=NNP(rq,sq)


            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,mV):
                jcb[0, 0] += dNVdr[k]*x[iconV[k,iel]]
                jcb[0, 1] += dNVdr[k]*y[iconV[k,iel]]
                jcb[1, 0] += dNVds[k]*x[iconV[k,iel]]
                jcb[1, 1] += dNVds[k]*y[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            rhoq=0.
            xq=0
            yq=0
            for k in range(0, mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                #rhoqV+=N[k]*rho[icon[k,iel]]
                dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.      ],
                                         [0.      ,dNVdy[i]],
                                         [dNVdy[i],dNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob
            #print(rhoq)
            # compute elemental rhs vector
            for i in range(0, mV):
                f_el[ndofV*i  ]+=NV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NV[i]*jcob*weightq*by(xq,yq)


            for i in range(0,mP):
                N_mat[0,i]=NP[i]
                N_mat[1,i]=NP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # end for jq
    # end for iq

    #-----------------------
    # compute total rhs
    #-----------------------

    v_el = np.array([u[iconV[0,iel]],v[iconV[0,iel]],\
                     u[iconV[1,iel]],v[iconV[1,iel]],\
                     u[iconV[2,iel]],v[iconV[2,iel]],\
                     u[iconV[3,iel]],v[iconV[3,iel]],\
                     u[iconV[4,iel]],v[iconV[4,iel]],\
                     u[iconV[5,iel]],v[iconV[5,iel]],\
                     u[iconV[6,iel]],v[iconV[6,iel]],\
                     u[iconV[7,iel]],v[iconV[7,iel]],\
                     u[iconV[8,iel]],v[iconV[8,iel]],\
                     u[iconV[9,iel]],v[iconV[9,iel]],\
                     u[iconV[10,iel]],v[iconV[10,iel]],\
                     u[iconV[11,iel]],v[iconV[11,iel]],\
                     u[iconV[12,iel]],v[iconV[12,iel]],\
                     u[iconV[13,iel]],v[iconV[13,iel]],\
                     u[iconV[14,iel]],v[iconV[14,iel]],\
                     u[iconV[15,iel]],v[iconV[15,iel]] ])

    # v_el = np.array([u[icon[0,iel]],v[icon[0,iel]],\
    #                  u[icon[1,iel]],v[icon[1,iel]],\
    #                  u[icon[2,iel]],v[icon[2,iel]],\
    #                  u[icon[3,iel]],v[icon[3,iel]] ])

    #print(f_el)

    p_el = np.array([p[iconP[0,iel]],\
                     p[iconP[1,iel]],\
                     p[iconP[2,iel]],\
                     p[iconP[3,iel]],\
                     p[iconP[4,iel]],\
                     p[iconP[5,iel]],\
                     p[iconP[6,iel]],\
                     p[iconP[7,iel]],\
                     p[iconP[8,iel]]])


    rhs_el=-f_el+K_el.dot(v_el)+G_el.dot(p_el)

    #-----------------------
    # assemble 
    #-----------------------

    #boundary 0-3 : x,y dofs
    for i in range(0,ndofV):
        idof0=2*iconV[0,iel]+i
        idof1=2*iconV[1,iel]+i
        idof2=2*iconV[2,iel]+i
        idof3=2*iconV[3,iel]+i
        if (bc_fix[idof0] and bc_fix[idof1]):  
           idofTr0=bc_nb[idof0]   
           idofTr1=bc_nb[idof1]
           idofTr2=bc_nb[idof2]
           idofTr3=bc_nb[idof3]
           rhs_cbf[idofTr0]+=rhs_el[0+i]   
           rhs_cbf[idofTr1]+=rhs_el[2+i]   
           rhs_cbf[idofTr2]+=rhs_el[4+i]   
           rhs_cbf[idofTr3]+=rhs_el[6+i]   
           M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hx
           M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hx
           M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hx
           M_prime[idofTr0,idofTr3]+=M_prime_el[0,3]*hx
           M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hx
           M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hx
           M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hx
           M_prime[idofTr1,idofTr3]+=M_prime_el[1,3]*hx
           M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hx
           M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hx
           M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hx
           M_prime[idofTr2,idofTr3]+=M_prime_el[2,3]*hx
           M_prime[idofTr3,idofTr0]+=M_prime_el[3,0]*hx
           M_prime[idofTr3,idofTr1]+=M_prime_el[3,1]*hx
           M_prime[idofTr3,idofTr2]+=M_prime_el[3,2]*hx
           M_prime[idofTr3,idofTr3]+=M_prime_el[3,3]*hx

           # print("0-3 ")
           # print("iel = "+str(iel))
           # print(idof0,idof1,idof2,idof3)


    #boundary 3-15 : x,y dofs
    for i in range(0,ndofV):
        idof0=2*iconV[3,iel]+i
        idof1=2*iconV[7,iel]+i
        idof2=2*iconV[11,iel]+i
        idof3=2*iconV[15,iel]+i
        if (bc_fix[idof0] and bc_fix[idof1]):  
           idofTr0=bc_nb[idof0]   
           idofTr1=bc_nb[idof1]
           idofTr2=bc_nb[idof2]
           idofTr3=bc_nb[idof3]
           rhs_cbf[idofTr0]+=rhs_el[6+i]   
           rhs_cbf[idofTr1]+=rhs_el[14+i]   
           rhs_cbf[idofTr2]+=rhs_el[22+i]   
           rhs_cbf[idofTr3]+=rhs_el[30+i]   
           M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hy
           M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hy
           M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hy
           M_prime[idofTr0,idofTr3]+=M_prime_el[0,3]*hy
           M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hy
           M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hy
           M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hy
           M_prime[idofTr1,idofTr3]+=M_prime_el[1,3]*hy
           M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hy
           M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hy
           M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hy
           M_prime[idofTr2,idofTr3]+=M_prime_el[2,3]*hy
           M_prime[idofTr3,idofTr0]+=M_prime_el[3,0]*hy
           M_prime[idofTr3,idofTr1]+=M_prime_el[3,1]*hy
           M_prime[idofTr3,idofTr2]+=M_prime_el[3,2]*hy
           M_prime[idofTr3,idofTr3]+=M_prime_el[3,3]*hy

           # print("3-15")
           # print("iel = "+str(iel))
           # print(idof0,idof1,idof2,idof3)

    #boundary 15-12 : x,y dofs
    for i in range(0,ndofV):
        idof0=2*iconV[15,iel]+i
        idof1=2*iconV[14,iel]+i
        idof2=2*iconV[13,iel]+i
        idof3=2*iconV[12,iel]+i
        if (bc_fix[idof0] and bc_fix[idof1]):  
           idofTr0=bc_nb[idof0]   
           idofTr1=bc_nb[idof1]
           idofTr2=bc_nb[idof2]
           idofTr3=bc_nb[idof3]
           rhs_cbf[idofTr0]+=rhs_el[30+i]   
           rhs_cbf[idofTr1]+=rhs_el[28+i]   
           rhs_cbf[idofTr2]+=rhs_el[26+i]   
           rhs_cbf[idofTr3]+=rhs_el[24+i]   
           M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hx
           M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hx
           M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hx
           M_prime[idofTr0,idofTr3]+=M_prime_el[0,3]*hx
           M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hx
           M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hx
           M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hx
           M_prime[idofTr1,idofTr3]+=M_prime_el[1,3]*hx
           M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hx
           M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hx
           M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hx
           M_prime[idofTr2,idofTr3]+=M_prime_el[2,3]*hx
           M_prime[idofTr3,idofTr0]+=M_prime_el[3,0]*hx
           M_prime[idofTr3,idofTr1]+=M_prime_el[3,1]*hx
           M_prime[idofTr3,idofTr2]+=M_prime_el[3,2]*hx
           M_prime[idofTr3,idofTr3]+=M_prime_el[3,3]*hx

           # print("15-12")
           # print("iel = "+str(iel))
           # print(idof0,idof1,idof2,idof3)

    #boundary 12-0 : x,y dofs
    for i in range(0,ndofV):
        idof0=2*iconV[12,iel]+i
        idof1=2*iconV[8,iel]+i
        idof2=2*iconV[4,iel]+i
        idof3=2*iconV[0,iel]+i
        if (bc_fix[idof0] and bc_fix[idof1]):  
           idofTr0=bc_nb[idof0]   
           idofTr1=bc_nb[idof1]
           idofTr2=bc_nb[idof2]
           idofTr3=bc_nb[idof3]
           rhs_cbf[idofTr0]+=rhs_el[24+i]   
           rhs_cbf[idofTr1]+=rhs_el[16+i]   
           rhs_cbf[idofTr2]+=rhs_el[8+i]   
           rhs_cbf[idofTr3]+=rhs_el[0+i]   
           M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hy
           M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hy
           M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hy
           M_prime[idofTr0,idofTr3]+=M_prime_el[0,3]*hy

           M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hy
           M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hy
           M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hy
           M_prime[idofTr1,idofTr3]+=M_prime_el[1,3]*hy

           M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hy
           M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hy
           M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hy
           M_prime[idofTr2,idofTr3]+=M_prime_el[2,3]*hy

           M_prime[idofTr3,idofTr0]+=M_prime_el[3,0]*hy
           M_prime[idofTr3,idofTr1]+=M_prime_el[3,1]*hy
           M_prime[idofTr3,idofTr2]+=M_prime_el[3,2]*hy
           M_prime[idofTr3,idofTr3]+=M_prime_el[3,3]*hy

           # print("12-0")
           # print("iel = "+str(iel))
           # print(idof0,idof1,idof2,idof3)

        #print(" ")


# end for iel

np.savetxt("rhs_cbf.ascii",rhs_cbf)
matfile=open("matrix.ascii","w")
for i in range(0,NfemTr):
   for j in range(0,NfemTr):
       if abs(M_prime[i,j])>1e-16:
          matfile.write(" %d %d %e \n " % (i,j,M_prime[i,j]))
matfile.close()

print("     -> M_prime (m,M) %.4e %.4e " %(np.min(M_prime),np.max(M_prime)))
print("     -> rhs_cbf (m,M) %.4e %.4e " %(np.min(rhs_cbf),np.max(rhs_cbf)))

print(type(M_prime))
print(M_prime.shape)


# for i in range(NfemTr):
#     print(i)
#     #print(bc_nb[i])
#     for j in range(NfemV):
#         if np.abs(i - bc_nb[j]) < eps:
#             print(j)
#     print(M_prime[i,i])
#     print(" ")

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

fig.savefig("tractions_CBF.pdf")




#####################################################################
# plot of solution
#####################################################################
# the 16-node Q3 element does not exist in vtk so it is split into 
# 9 Q1 elements for visualisation purposes only.

nel2=(nnx-1)*(nny-1)
iconV2=np.zeros((4,nel2),dtype=np.int16)

counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconV2[0,counter]=i+j*nnx
        iconV2[1,counter]=i+1+j*nnx
        iconV2[2,counter]=i+1+(j+1)*nnx
        iconV2[3,counter]=i+(j+1)*nnx
        counter += 1

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel2))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %pth(x[i],y[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel2):
    vtufile.write("%d %d %d %d\n" %(iconV2[0,iel],iconV2[1,iel],iconV2[2,iel],iconV2[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel2):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel2):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")