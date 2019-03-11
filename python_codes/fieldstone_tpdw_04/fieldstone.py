import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt

#This version of fieldstone is designed to sink a ball and see what happens with the tractions

#------------------------------------------------------------------------------

def rho(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def mu(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=1.e2
    else:
       val=1.
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


#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

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
   nelx = 64
   nely = 64
   visu = 1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs


pnormalise=False
sparse=False

gx=0
gy=1

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
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,1]=True
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,2]=True
    if y[i]>(Ly-eps):
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,3]=True



print("setup: boundary conditions: %.3f s" % (time.time() - start))

NfemTr=np.sum(bc_fix)

bc_nb=np.zeros(NfemV,dtype=np.int32)  # boundary condition, yes/no

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

if sparse:
  if pnormalise:
    A_sparse = lil_matrix((Nfem+1,Nfem+1))
  else:
    A_sparse = lil_matrix((Nfem,Nfem))
else:
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
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*mu(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*rho(xq,yq)*gx
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*rho(xq,yq)*gy
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
                    if sparse:
                      A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                      K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            if sparse:
              A_sparse[m1,NfemV+iel]+=G_el[ikk,0]
              A_sparse[NfemV+iel,m1]+=G_el[ikk,0]
            else:  
              G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if sparse:
  if pnormalise:
    rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
    A_sparse[Nfem,NfemV:Nfem]=1
    A_sparse[NfemV:Nfem,Nfem]=1
  else:
    rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
else:

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
# compute strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e = np.zeros(nel,dtype=np.float64)  

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

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

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

sxxn1=exxn1+q1
syyn1=eyyn1+q1
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
    sigmayy_el[iel]=(-p[iel]+2.*mu(x[i],y[i])*eyy[iel])

# np.savetxt('sigmayy_el.ascii',np.array([xc[nel-nelx:nel],\
#                                         (sigmayy_el[nel-nelx:nel])/hx,\
#                                        ]).T,header='# xc,sigmayy')

# np.savetxt('sigmayy_C-N.ascii',np.array([x[nnp-nnx:nnp],\
#                                         (-q1[nnp-nnx:nnp]+2.*mu(x[nnp-nnx:nnp],y[nnp-nnx:nnp])*eyyn1[nnp-nnx:nnp])/hx,\
#                                         ]).T,header='# x,sigmayy')

# np.savetxt('sigmayy_LS.ascii',np.array([x[nnp-nnx:nnp],\
#                                         (-q3[nnp-nnx:nnp]+2.*mu(x[nnp-nnx:nnp],y[nnp-nnx:nnp])*eyyn3[nnp-nnx:nnp])/hx,\
#                                         ]).T,header='# x,sigmayy')





# print("     -> sigmayy_el       (N-E) %6f " % ((sigmayy_el[nel-1])/hx) ) 
# print("     -> sigmayy_nod C->N (N-E) %6f " % ((-q1[nnp-1]+2.*mu(x[nnp-1],y[nnp-1])*eyyn1[nnp-1])/hx) ) 
# print("     -> sigmayy_nod LS   (N-E) %6f " % ((-q3[nnp-1]+2.*mu(x[nnp-1],y[nnp-1])*eyyn3[nnp-1])/hx) )

#####################################################################
# Consistent Boundary Flux method
#####################################################################


M_prime = np.zeros((NfemTr,NfemTr),np.float64)
rhs_cbf = np.zeros(NfemTr,np.float64)
tx = np.zeros(nnp,np.float64)
ty = np.zeros(nnp,np.float64)

# M_prime_el =(hx/2.)*np.array([ \
# [2./3.,1./3.],\
# [1./3.,2./3.]])

M_prime_el =(hx/2.)*np.array([ \
[1,0],\
[0,1]])


CBF_use_smoothed_pressure=False

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
            for k in range(0, m):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*mu(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*rho(xq,yq)*gx
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*rho(xq,yq)*gy
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
    if CBF_use_smoothed_pressure:
      rhs_el=-f_el+K_el.dot(v_el)+G_el[:,0]*p_smoothed[iel]
    else:
      rhs_el=-f_el+K_el.dot(v_el)+G_el[:,0]*p[iel]

    use_theoretical_pressure=False
    if use_theoretical_pressure:
      rhs_el=-f_el+K_el.dot(v_el)+G_el[:,0]*pressure(x[icon[k,iel]]+hx/2.0,y[icon[k,iel]]+hy/2.0)


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

np.savetxt('sigmayy_cbf.ascii',np.array([x[nnp-nnx:nnp],ty[nnp-nnx:nnp]]).T,header='# x,sigmayy')

print("     -> tx (m,M) %.4e %.4e " %(np.min(tx),np.max(tx)))
print("     -> ty (m,M) %.4e %.4e " %(np.min(ty),np.max(ty)))

np.savetxt('tractions.ascii',np.array([x,y,tx,ty]).T,header='# x,y,tx,ty')

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))

#ax1 contains the lower tractions I guess

ax1.plot(-exyn1[:nnx],label="$t_x$ (C->N)")
ax1.plot(-eyyn1[:nnx]-q1[:nnx],label="$t_y$ (C->N)")
ax1.plot(-tx[:nnx],label="$t_x$ (CBF)")
ax1.plot(-ty[:nnx],label="$t_y$ (CBF)")
ax1.legend()
ax1.set_title("Lower Boundary")

ax2.plot(exyn1[(nnp-nnx):],label="$t_x$ (C->N)")
ax2.plot(eyyn1[(nnp-nnx):]+q1[(nnp-nnx):],label="$t_y$ (C->N)")
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
    left_tx_cn.append(exxn1[i]+q1[i])
    left_ty_cn.append(exyn1[i])
    left_y.append(y[i])
  if (Lx-x[i])<eps:
    right_tx_cn.append(exxn1[i]+q1[i])
    right_ty_cn.append(exyn1[i])
    right_y.append(y[i])

right_y=np.array(right_y)
left_y=np.array(left_y)
left_tx_cn=np.array(left_tx_cn)
right_tx_cn=np.array(right_tx_cn)
left_ty_cn=np.array(left_ty_cn)
right_ty_cn=np.array(right_ty_cn)

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

fig.savefig("tractions_CN_CBF.pdf")


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
vtufile.write("<DataArray type='Float32' Name='syy (C-N)' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %syyn1[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='sxy (C-N)' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %sxyn1[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %rho(x[i],y[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='mu' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %mu(x[i],y[i]))
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

# #####################################################################
# # plot of solution
# #####################################################################

# u_temp=np.reshape(u,(nny,nnx))
# v_temp=np.reshape(v,(nny,nnx))
# q_temp=np.reshape(q,(nny,nnx))
# p_temp=np.reshape(p,(nely,nelx))
# e_temp=np.reshape(e,(nely,nelx))
# exx_temp=np.reshape(exx,(nely,nelx))
# eyy_temp=np.reshape(eyy,(nely,nelx))
# exy_temp=np.reshape(exy,(nely,nelx))
# error_u_temp=np.reshape(error_u,(nny,nnx))
# error_v_temp=np.reshape(error_v,(nny,nnx))
# error_q_temp=np.reshape(error_q,(nny,nnx))
# error_p_temp=np.reshape(error_p,(nely,nelx))

# fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(18,18))

# uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
# pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

# onePlot(u_temp,       0, 0, "$v_x$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
# onePlot(v_temp,       0, 1, "$v_y$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
# onePlot(p_temp,       0, 2, "$p$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
# onePlot(q_temp,       0, 3, "$q$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
# onePlot(exx_temp,     1, 0, "$\dot{\epsilon}_{xx}$", "x", "y", pextent, Lx, Ly, 'viridis')
# onePlot(eyy_temp,     1, 1, "$\dot{\epsilon}_{yy}$", "x", "y", pextent, Lx, Ly, 'viridis')
# onePlot(exy_temp,     1, 2, "$\dot{\epsilon}_{xy}$", "x", "y", pextent, Lx, Ly, 'viridis')
# onePlot(e_temp,       1, 3, "$\dot{\epsilon}$",      "x", "y", pextent, Lx, Ly, 'viridis')
# onePlot(error_u_temp, 2, 0, "$v_x-t^{th}_x$",        "x", "y", uextent,  0,  0, 'Spectral_r')
# onePlot(error_v_temp, 2, 1, "$v_y-t^{th}_y$",        "x", "y", uextent,  0,  0, 'Spectral_r')
# onePlot(error_p_temp, 2, 2, "$p-p^{th}$",            "x", "y", uextent,  0,  0, 'RdGy_r')
# onePlot(error_q_temp, 2, 3, "$q-p^{th}$",            "x", "y", uextent,  0,  0, 'RdGy_r')

# plt.subplots_adjust(hspace=0.5)
# plt.savefig('solution.pdf', bbox_inches='tight')


# ##############################################################
# #2D plots of the strain rates and theoretical strain rates
# ##############################################################
# fig,((ax1,ax2,ax12),(ax3,ax4,ax34),(ax5,ax6,ax56)) = plt.subplots(3,3,figsize=(10,10))
# X_grid,Y_grid=np.meshgrid(np.linspace(0,Lx,nnx),np.linspace(0,Ly,nny))
# ax1.contourf(sigma_xx(X_grid,Y_grid))
# fig.colorbar(ax1.contourf(sigma_xx(X_grid,Y_grid)),ax=ax1)
# ax2.contourf(np.reshape(exxn1+q1,(nny,nnx)))
# fig.colorbar(ax2.contourf(np.reshape(exxn1+q1,(nny,nnx))),ax=ax2)
# ax12.contourf(sigma_xx(X_grid,Y_grid)-np.reshape(exxn1+q1,(nny,nnx)))
# fig.colorbar(ax12.contourf(sigma_xx(X_grid,Y_grid)-np.reshape(exxn1+q1,(nny,nnx))),ax=ax12)

# fig.savefig("sigma_contours.pdf")



print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


