import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def density(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0)) 
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

def viscosity(T,gamma_T):#,gamma_y,mu_star):
    val=np.exp(-gamma_T*T)
    val=min(2.0,val)
    val=max(1.e-5,val)
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

ndim=2  # number of physical dimensions
m=4     # number of nodes making up an element
ndof=2  # number of V degrees of freedom per node
ndofT=1 # number of T degrees of freedom per node

if int(len(sys.argv) == 3):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelr = 16
   visu = 1

assert (nelr>0.), "nnx should be positive" 

R1=1.22
R2=2.22

dr=(R2-R1)/nelr
area=np.pi*(R2**2-R1**2)
nelt=int(2.*math.pi*R2/dr)
nel=nelr*nelt  # number of elements, total


nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes
NfemV=nnp*ndof  # Total number of V degrees of freedom
NfemT=nnp*ndofT # Total number of T degrees of freedom

penalty=1.e4 
rho0=1.
alpha=1e-4
Ra=1e2
g0=Ra/alpha
T0= 0.
CFL_nb=1.
Temp_surf=0
Temp_cmb=1
gamma_T=np.log(1e5)
gamma_y=np.log(1e1)
mu_star=1e-3
N0=3
hcapa=1.
hcond=1.
nstep=20
use_BA=True
use_EBA=False

eps=1.e-10
sqrt3=np.sqrt(3.)

if use_BA:
   use_shearheating=False
   use_adiabatic_heating=False

if use_EBA:
   use_shearheating=True
   use_adiabatic_heating=True

#################################################################

model_time=np.zeros(nstep,dtype=np.float64)
vrms=np.zeros(nstep,dtype=np.float64)
Nu=np.zeros(nstep,dtype=np.float64)
Tavrg=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
visc_diss=np.zeros(nstep,dtype=np.float64)
work_grav=np.zeros(nstep,dtype=np.float64)
EK=np.zeros(nstep,dtype=np.float64)
EG=np.zeros(nstep,dtype=np.float64)
ET=np.zeros(nstep,dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)
heatflux_boundary=np.zeros(nstep,dtype=np.float64)
adiab_heating=np.zeros(nstep,dtype=np.float64)

#################################################################
# grid point setup
#################################################################

print("grid point setup")

x=np.empty(nnp,dtype=np.float64)     # x coordinates
y=np.empty(nnp,dtype=np.float64)     # y coordinates
r=np.empty(nnp,dtype=np.float64)     # cylindrical coordinate r 
theta=np.empty(nnp,dtype=np.float64) # cylindrical coordinate theta 

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sy = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        x[counter]=i*sx
        y[counter]=j*sy
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=x[counter]
        yi=y[counter]
        t=xi/Louter*2.*math.pi    
        x[counter]=math.cos(t)*(R1+yi)
        y[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(y[counter],x[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        icon[0, counter] = icon2 
        icon[1, counter] = icon1
        icon[2, counter] = icon4
        icon[3, counter] = icon3
        counter += 1

print("connectivity (%.3fs)" % (time.time() - start))

#################################################################
# define velocity boundary conditions
#################################################################
start = time.time()

bc_fixV = np.zeros(NfemV, dtype=np.bool)  
bc_valV = np.zeros(NfemV, dtype=np.float64) 

for i in range(0, nnp):
    if r[i]<R1+eps:
       bc_fixV[i*ndof]   = True ; bc_valV[i*ndof]   = 0.
       bc_fixV[i*ndof+1] = True ; bc_valV[i*ndof+1] = 0. 
    if r[i]>(R2-eps):
       bc_fixV[i*ndof]   = True ; bc_valV[i*ndof]   = 0. 
       bc_fixV[i*ndof+1] = True ; bc_valV[i*ndof+1] = 0.

print("defining V boundary conditions (%.3fs)" % (time.time() - start))

#################################################################
# define temperature boundary conditions
#################################################################
start = time.time()

bc_fixT = np.zeros(NfemT, dtype=np.bool)  
bc_valT = np.zeros(NfemT, dtype=np.float64) 

for i in range(0, nnp):
    if r[i]<R1+eps:
       bc_fixT[i] = True ; bc_valT[i] = Temp_cmb
    if r[i]>(R2-eps):
       bc_fixT[i] = True ; bc_valT[i] = Temp_surf

print("defining T boundary conditions (%.3fs)" % (time.time() - start))

#################################################################
# initial temperature field 
#################################################################
start = time.time()

T     = np.zeros(nnp,dtype=np.float64)          # temperature 

for i in range(0,nnp):
    s=(R2-r[i])/(R2-R1)
    T[i]=s+0.2*s*(1.-s)*np.cos(N0*theta[i])

print("temperature layout (%.3fs)" % (time.time() - start))


#################################################################
# nodal density and viscosity setup
#################################################################
start = time.time()

rho=np.zeros(nnp,dtype=np.float64)
rho_prev=np.zeros(nnp,dtype=np.float64)
drhodt=np.zeros(nnp,dtype=np.float64)
eta=np.zeros(nnp,dtype=np.float64)

for i in range(0,nnp):
    rho[i]=density(rho0,alpha,T[i],T0)
    eta[i]=viscosity(T[i],gamma_T)

rho_prev[:]=rho[:]

print("setup: T,rho: %.3f s" % (time.time() - start))

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
q     = np.zeros(nnp,dtype=np.float64)          # nodal pressure 
q_prev= np.zeros(nnp,dtype=np.float64)
dqdt  = np.zeros(nnp,dtype=np.float64)

for istep in range(0,nstep):
    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

    #################################################################
    # build FE matrix
    #################################################################
    start = time.time()

    a_mat = np.zeros((NfemV,NfemV),dtype=np.float64)  # matrix of Ax=b
    b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
    rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side of Ax=b
    N     = np.zeros(m,dtype=np.float64)            # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el = np.zeros(m * ndof)
        a_el = np.zeros((m * ndof, m * ndof), dtype=np.float64)

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
                Tq=0.0
                etaq=0.0
                rhoq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                    etaq+=N[k]*eta[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                # compute elemental a_mat matrix
                a_el += b_mat.T.dot(c_mat.dot(b_mat))*etaq*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[2*i  ]+=N[i]*jcob*wq*gx(xq,yq,g0)*rhoq
                    b_el[2*i+1]+=N[i]*jcob*wq*gy(xq,yq,g0)*rhoq

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

        # apply boundary conditions
        for k1 in range(0,m):
            for i1 in range(0,ndof):
                m1 =ndof*icon[k1,iel]+i1
                if bc_fixV[m1]: 
                   fixt=bc_valV[m1]
                   ikk=ndof*k1+i1
                   aref=a_el[ikk,ikk]
                   for jkk in range(0,m*ndof):
                       b_el[jkk]-=a_el[jkk,ikk]*fixt
                       a_el[ikk,jkk]=0.
                       a_el[jkk,ikk]=0.
                   a_el[ikk,ikk]=aref
                   b_el[ikk]=aref*fixt

        # assemble matrix a_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndof):
                ikk=ndof*k1          +i1
                m1 =ndof*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndof):
                        jkk=ndof*k2          +i2
                        m2 =ndof*icon[k2,iel]+i2
                        a_mat[m1,m2]+=a_el[ikk,jkk]
                rhs[m1]+=b_el[ikk]

    print("build FE matrixs & rhs (%.3fs)" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
    print("solving system (%.3fs)" % (time.time() - start))

    #####################################################################
    # put solution into separate x,y velocity arrays
    #####################################################################
    start = time.time()

    u,v=np.reshape(sol,(nnp,2)).T

    print("u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    u_stats[istep,0]=np.min(u) ; u_stats[istep,1]=np.max(u)
    v_stats[istep,0]=np.min(v) ; v_stats[istep,1]=np.max(v)

    np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("reshape solution (%.3fs)" % (time.time() - start))

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

        N[0:m]=NNV(rq,sq)
        dNdr[0:m]=dNNVdr(rq,sq)
        dNds[0:m]=dNNVds(rq,sq)

        jcb=np.zeros((2,2),dtype=float)
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

        for k in range(0, m):
            xc[iel] += N[k]*x[icon[k,iel]]
            yc[iel] += N[k]*y[icon[k,iel]]
            exx[iel] += dNdx[k]*u[icon[k,iel]]
            eyy[iel] += dNdy[k]*v[icon[k,iel]]
            exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+\
                        0.5*dNdx[k]*v[icon[k,iel]]

        p[iel]=-penalty*(exx[iel]+eyy[iel])

    print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
    print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
    np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute p & sr | time: %.3f s" % (time.time() - start))

    ######################################################################
    # compute time step value 
    ######################################################################
    start = time.time()

    dt1=CFL_nb*min(sx,sy)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL_nb*min(sx,sy)**2/(hcond/hcapa/rho0)

    dt=min(dt1,dt2)

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    dt_stats[istep]=dt

    print('     -> dt1= %.6e dt2= %.6e dt= %.6e' % (dt1,dt2,dt))

    print("compute timestep: %.3f s" % (time.time() - start))

    ######################################################################
    # compute nodal pressure
    ######################################################################
    start = time.time()

    count=np.zeros(nnp,dtype=np.float64)
    q[:]=0

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

    dqdt=(q[:]-q_prev[:])/dt

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute q: %.3f s" % (time.time() - start))

    ######################################################################
    # compute vrms 
    # QUESTION: should i divide by area or by int 1 dV ?
    ######################################################################
    start = time.time()

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
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                uq=0.
                vq=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                rhoq=0.
                etaq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                    etaq+=N[k]*eta[icon[k,iel]]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=0.5*dNdy[k]*u[icon[k,iel]]+\
                          0.5*dNdx[k]*v[icon[k,iel]]
                vrms[istep]+=(uq**2+vq**2)*weightq*jcob
                visc_diss[istep]+=2.*etaq*(exxq**2+eyyq**2+2*exyq**2)*weightq*jcob
                EK[istep]+=0.5*rhoq*(uq**2+vq**2)*weightq*jcob
                EG[istep]+=rhoq*(gx(xq,yq,g0)*xq+gy(xq,yq,g0)*yq)*weightq*jcob
                work_grav[istep]+=(rhoq-rho0)*(gx(xq,yq,g0)*uq+gy(xq,yq,g0)*vq)*weightq*jcob

    vrms[istep]=np.sqrt(vrms[istep]/area)

    print("     -> vrms= %.6e ; Ra= %.6e ; vrmsdiff= %.6e " % (vrms[istep],Ra,vrms[istep]-vrms[0]))

    print("compute vrms, EK, EG, WAG: %.3f s" % (time.time() - start))

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    # ToDo: look at np.outer product in python so N_mat -> N

    start = time.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(m,dtype=np.float64)

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)
        f_el=np.zeros(m*ndofT,dtype=np.float64)

        for k in range(0,m):
            Tvect[k]=T[icon[k,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N_mat[0,0]=0.25*(1.-rq)*(1.-sq)
                N_mat[1,0]=0.25*(1.+rq)*(1.-sq)
                N_mat[2,0]=0.25*(1.+rq)*(1.+sq)
                N_mat[3,0]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

                # calculate jacobian matrix
                jcb=np.zeros((2, 2),dtype=float)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy and Phi
                Tq=0.
                rhoq=0.
                etaq=0.
                vel[0,0]=0.
                vel[0,1]=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                dqdxq=0.
                dqdyq=0.
                for k in range(0,m):
                    Tq+=N_mat[k,0]*T[icon[k,iel]]
                    rhoq+=N_mat[k,0]*rho[icon[k,iel]]
                    etaq+=N_mat[k,0]*eta[icon[k,iel]]
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])*.5
                    dqdxq+=dNdx[k]*q[icon[k,iel]]
                    dqdyq+=dNdy[k]*q[icon[k,iel]]
                Phiq=2.*etaq*(exxq**2+eyyq**2+2*exyq**2)

                if use_BA or use_EBA:
                   rho_lhs=rho0
                else:
                   rho_lhs=rhoq
               # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho_lhs*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho_lhs*hcapa*wq*jcob

                # compute shear heating rhs term
                if use_shearheating:
                   f_el[:]+=N_mat[:,0]*Phiq*jcob*wq
                else:
                   f_el[:]+=0

                # compute adiabatic heating rhs term 
                if use_adiabatic_heating:
                   f_el[:]+=N_mat[:,0]*alpha*Tq*(vel[0,0]*dqdxq+vel[0,1]*dqdyq)*wq*jcob
                else:
                   f_el[:]+=0


                a_el=MM+(Ka+Kd)*dt

                b_el=MM.dot(Tvect) + f_el*dt

                # apply boundary conditions

                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    if bc_fixT[m1]:
                       Aref=a_el[k1,k1]
                       for k2 in range(0,m):
                           m2=icon[k2,iel]
                           b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                           a_el[k1,k2]=0
                           a_el[k2,k1]=0
                       a_el[k1,k1]=Aref
                       b_el[k1]=Aref*bc_valT[m1]

               # assemble matrix A_mat and right hand side rhs
                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    for k2 in range(0,m):
                        m2=icon[k2,iel]
                        A_mat[m1,m2]+=a_el[k1,k2]
                    rhs[m1]+=b_el[k1]

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix T: %.3f s" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats[istep,0]=np.min(T) ; T_stats[istep,1]=np.max(T)

    print("solve T: %.3f s" % (time.time() - start))

    #####################################################################
    # update density 
    #####################################################################
    start = time.time()

    for i in range(0,nnp):
        rho[i]=density(rho0,alpha,T[i],T0)
        eta[i]=viscosity(T[i],gamma_T)

    drhodt=(rho[:]-rho_prev[:])/dt

    print("     -> rho (m,M) %.4f %.4f " %(np.min(rho),np.max(rho)))

    print("compute density: %.3f s" % (time.time() - start))

    #####################################################################
    # compute average of temperature
    #####################################################################
    start = time.time()

    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                Tq=0.
                uq=0.
                vq=0.
                rhoq=0.
                dqdxq=0.
                dqdyq=0.
                for k in range(0,m):
                    Tq+=N[k]*T[icon[k,iel]]
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                    dqdxq+=dNdx[k]*q[icon[k,iel]]
                    dqdyq+=dNdy[k]*q[icon[k,iel]]
                Tavrg[istep]+=Tq*wq*jcob
                if use_BA or use_EBA:
                   ET[istep]+=rho0*hcapa*Tq*wq*jcob
                else:
                   ET[istep]+=rhoq*hcapa*Tq*wq*jcob
                adiab_heating[istep]+=alpha*Tq*(uq*dqdxq+vq*dqdyq)*wq*jcob

    Tavrg[istep]/=area

    print("     -> avrg T= %.6e" % Tavrg[istep])

    print("compute <T>: %.3f s" % (time.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = time.time()

    if visu==1 or istep%1==0:
       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%f\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f %10f %10f \n" %(gx(x[i],y[i],g0),gy(x[i],y[i],g0),0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='T' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %r[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %theta[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %rho[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10f \n" %eta[i])
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
       print("export to vtu: %.3f s" % (time.time() - start))

    #####################################################################
    # write to file 
    #####################################################################
    start = time.time()

    np.savetxt('vrms.ascii',np.array([model_time[0:istep],vrms[0:istep]]).T,header='# t,vrms')
    np.savetxt('Nu.ascii',np.array([model_time[0:istep],Nu[0:istep]]).T,header='# t,Nu')
    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep],Tavrg[0:istep]]).T,header='# t,Tavrg')
    np.savetxt('EK.ascii',np.array([model_time[0:istep],EK[0:istep]]).T,header='# t,EK')
    np.savetxt('EG.ascii',np.array([model_time[0:istep],EG[0:istep],EG[0:istep]-EG[0]]).T,header='# t,EG,EG-EG(0)')
    np.savetxt('ET.ascii',np.array([model_time[0:istep],ET[0:istep]]).T,header='# t,ET')
    np.savetxt('viscous_dissipation.ascii',np.array([model_time[0:istep],visc_diss[0:istep]]).T,header='# t,Phi')
    np.savetxt('work_grav.ascii',np.array([model_time[0:istep],work_grav[0:istep]]).T,header='# t,W')
    np.savetxt('heat_flux_boundary.ascii',np.array([model_time[0:istep],heatflux_boundary[0:istep]]).T,header='# t,q')
    np.savetxt('adiabatic_heating.ascii',np.array([model_time[0:istep],adiab_heating[0:istep]]).T,header='# t,ad.heat.')
    np.savetxt('viscous_dissipation_adim.ascii',np.array([model_time[0:istep],visc_diss[0:istep]]).T,header='# t,Phi')
    np.savetxt('work_grav_adim.ascii',np.array([model_time[0:istep],work_grav[0:istep]]).T,header='# t,W')
    np.savetxt('conservation.ascii',np.array([model_time[0:istep],visc_diss[0:istep],\
                                    adiab_heating[0:istep],heatflux_boundary[0:istep]]).T,header='# t,W')
    np.savetxt('u_stats.ascii',np.array([model_time[0:istep],u_stats[0:istep,0],u_stats[0:istep,1]]).T,header='# t,min(u),max(u)')
    np.savetxt('v_stats.ascii',np.array([model_time[0:istep],v_stats[0:istep,0],v_stats[0:istep,1]]).T,header='# t,min(v),max(v)')
    np.savetxt('T_stats.ascii',np.array([model_time[0:istep],T_stats[0:istep,0],T_stats[0:istep,1]]).T,header='# t,min(T),max(T)')
    np.savetxt('dt_stats.ascii',np.array([model_time[0:istep],dt_stats[0:istep]]).T,header='# t,dt')

    if istep>0:
       np.savetxt('dETdt.ascii',np.array([model_time[1:istep],  ((ET[1:istep]-ET[0:istep-1])/dt)   ]).T,header='# t,ET')

    print("output stats: %.3f s" % (time.time() - start))

    #####################################################################

    q_prev[:]=q[:]
    rho_prev[:]=rho[:]

################################################################################################
# END OF TIMESTEPPING
################################################################################################

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")