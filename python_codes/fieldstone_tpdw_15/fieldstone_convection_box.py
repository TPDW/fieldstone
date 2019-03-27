import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
import tkinter

#------------------------------------------------------------------------------

def rho(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0))
    return val

def mu(T):
    val=1.
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------SimpleFEM----------")
print("-----------------------------")

sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofV=2      # number of degrees of freedom per node
ndofT=1      # number of degrees of freedom per node
Lx=1.        # horizontal extent of the domain 
Ly=1.        # vertical extent of the domain 
Ra=1e5       # Rayleigh number
alpha=1e-2   # thermal expansion coefficient
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density
T0=0         # reference temperature
CFL=1.       # CFL number 
gy=-Ra/alpha # vertical component of gravity vector
penalty=1.e7 # penalty coefficient value
nstep=2000   # maximum number of timestep   
tol=2e-5

Nu_prev = 0
vrms_prev=0


# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
else:
   nelx = 64
   nely = 64

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV  # Total number of degrees of velocity freedom
NfemT=nnp*ndofT  # Total number of degrees of temperature freedom

#####################################################################
# grid point setup 
#####################################################################

print("grid point setup")

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates
counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1

#####################################################################
# connectivity
#####################################################################

print("connectivity array")

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

#####################################################################
# define velocity boundary conditions
#####################################################################

print("defining velocity boundary conditions")

bc_fixV=np.zeros(NfemV,dtype=np.bool) 
bc_valV=np.zeros(NfemV,dtype=np.float64) 

for i in range(0,nnp):
    if x[i]<eps:
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if x[i]>(Lx-eps):
       bc_fixV[i*ndofV]   = True ; bc_valV[i*ndofV]   = 0.
    if y[i]<eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0.

#####################################################################
# define temperature boundary conditions
#####################################################################

print("defining temperature boundary conditions")

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 


for i in range(0,nnp):
    if y[i]<eps:
       bc_fixT[i]=True ; bc_valT[i]=1.
    if y[i]>(Ly-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.

NfemTCBF=sum(bc_fixT)

bc_nbT=np.zeros(NfemT,dtype=np.int32)  # boundary condition, yes/no

counter=0
for i in range(0,NfemT):
  if (bc_fixT[i]):
     bc_nbT[i]=counter
     counter+=1

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(nnp,dtype=np.float64)

for i in range(0,nnp):
    T[i]=1.-y[i]-0.01*math.cos(math.pi*x[i])*math.sin(math.pi*y[i])

np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

#####################################################################
# create necessary arrays 
#####################################################################

N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)  # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)  # y-component velocity
Tvect = np.zeros(4,dtype=np.float64)   
k_mat = np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]],dtype=np.float64) 
c_mat = np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
time=np.zeros(nstep,dtype=np.float64) 
Nusselt=np.zeros(nstep,dtype=np.float64) 

#==============================================================================
# time stepping loop
#==============================================================================

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)

#####################################################################
# build FE matrix
#####################################################################

    print("-----------------------------")
    print("building Stokes matrix and rhs")

    A_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemV,dtype=np.float64)         # FE rhs 
    B_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix 

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el=np.zeros(m*ndofV,dtype=np.float64)
        a_el=np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

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
                jcb = np.zeros((2, 2),dtype=np.float64)
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
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

                # construct 3x8 B_mat matrix
                for i in range(0, m):
                    B_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                # compute elemental A_mat matrix
                a_el += B_mat.T.dot(c_mat.dot(B_mat))*mu(Tq)*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[2*i+1]+=N[i]*jcob*wq*rho(rho0,alpha,Tq,T0)*gy

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
            B_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                  [0.     ,dNdy[i]],
                                  [dNdy[i],dNdx[i]]]

        # compute elemental matrix
        a_el += B_mat.T.dot(k_mat.dot(B_mat))*penalty*wq*jcob

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*icon[k2,iel]+i2
                        A_mat[m1,m2]+=a_el[ikk,jkk]
                rhs[m1]+=b_el[ikk]

    #################################################################
    # impose boundary conditions
    #################################################################

    print("imposing boundary conditions")

    for i in range(0,NfemV):
        if bc_fixV[i]:
           A_matref = A_mat[i,i]
           for j in range(0,NfemV):
               rhs[j]-= A_mat[i,j]*bc_valV[i]
               A_mat[i,j]=0.
               A_mat[j,i]=0.
               A_mat[i,i]=A_matref
           rhs[i]=A_matref*bc_valV[i]

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    sol = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve V time: %.3f s" % (timing.time() - start))

    #################################################################
    # put solution into separate x,y velocity arrays
    #################################################################

    u,v=np.reshape(sol,(nnp,2)).T

    print("u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    #################################################################
    # compute timestep
    #################################################################

    dt1=CFL*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL*(Lx/nelx)**2/(hcond/hcapa/rho0)

    dt=np.min([dt1,dt2])

    if istep==0:
       time[istep]=dt
    else:
       time[istep]=time[istep-1]+dt

    print('dt1= %.6f' %dt1)
    print('dt2= %.6f' %dt2)
    print('dt = %.6f' %dt)

    #################################################################
    # build temperature matrix
    #################################################################

    t_prev=T

    print("-----------------------------")
    print("building temperature matrix and rhs")

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

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

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                a_el=MM+(Ka+Kd)*dt

                b_el=MM.dot(Tvect)

                # assemble matrix A_mat and right hand side rhs
                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    for k2 in range(0,m):
                        m2=icon[k2,iel]
                        A_mat[m1,m2]+=a_el[k1,k2]
                    rhs[m1]+=b_el[k1]

    #################################################################
    # apply boundary conditions
    #################################################################

    print("imposing boundary conditions temperature")

    for i in range(0,NfemT):
        if bc_fixT[i]:
           A_matref = A_mat[i,i]
           for j in range(0,NfemT):
               rhs[j]-= A_mat[i, j] * bc_valT[i]
               A_mat[i,j]=0.
               A_mat[j,i]=0.
               A_mat[i,i] = A_matref
           rhs[i]=A_matref*bc_valT[i]

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve T time: %.3f s" % (timing.time() - start))

    print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    #################################################################
    # compute vrms 
    #################################################################

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
                jcob = np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                uq=0.
                vq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                vrms[istep]+=(uq**2+vq**2)*wq*jcob

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))

    print("time= %.6f ; vrms   = %.6f" %(time[istep],vrms[istep]))

    #################################################################
    # compute Nusselt number at top
    #################################################################

    for iel in range(0,nel):
        qy=0.
        rq=0.
        sq=0.
        wq=2.*2.
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
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        for k in range(0,m):
            qy+=-hcond*dNdy[k]*T[icon[k,iel]]
        if y[icon[3,iel]]>Ly-eps:
           Nusselt[istep]+=qy*hx

    print("time= %.6f ; Nusselt= %.6f" %(time[istep],Nusselt[istep]))

    ################################################################
    ####### CBF
    ################################################################


    M_prime = np.zeros((NfemTCBF,NfemTCBF),np.float64)
    rhs_cbf = np.zeros(NfemTCBF,np.float64)
    dTdy_CBF = np.zeros(nnp,np.float64)

    M_prime_el =(hx/2.)*np.array([ \
    [2./3.,1./3.],\
    [1./3.,2./3.]])

    # print("test test fuck")

    for iel in range(nel):

        T_el = np.array([T[icon[0,iel]],T[icon[1,iel]],T[icon[2,iel]],T[icon[3,iel]]])
        T_prev_el = np.array([t_prev[icon[0,iel]],t_prev[icon[1,iel]],t_prev[icon[2,iel]],t_prev[icon[3,iel]]])

        T_el_dot = (T_el - T_prev_el)/dt

        rhs_el=np.zeros(m*ndofT,dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 

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

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                rhs_el += -MM.dot(T_el_dot) - (Ka+Kd).dot(T_el)

      #-----------------------
      # assemble 
      #-----------------------

        #boundary 0-1 : x,y dofs
        idof0=icon[0,iel]
        idof1=icon[1,iel]
        # print("what the fuck")
        # print(idof0)
        # print(idof1)
        if (bc_fixT[idof0] and bc_fixT[idof1]):  
             idofTr0=bc_nbT[idof0]   
             idofTr1=bc_nbT[idof1]
             rhs_cbf[idofTr0]+=rhs_el[0]   
             rhs_cbf[idofTr1]+=rhs_el[1]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

        #boundary 1-2 : x,y dofs
        idof0=icon[1,iel]
        idof1=icon[2,iel]
        if (bc_fixT[idof0] and bc_fixT[idof1]):  
             idofTr0=bc_nbT[idof0]   
             idofTr1=bc_nbT[idof1]
             rhs_cbf[idofTr0]+=rhs_el[1]   
             rhs_cbf[idofTr1]+=rhs_el[2]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

        #boundary 2-3 : x,y dofs
        idof0=icon[2,iel]
        idof1=icon[3,iel]
        # print(idof0)
        # print(idof1)
        if (bc_fixT[idof0] and bc_fixT[idof1]):  
             idofTr0=bc_nbT[idof0]   
             idofTr1=bc_nbT[idof1]
             rhs_cbf[idofTr0]+=rhs_el[2]   
             rhs_cbf[idofTr1]+=rhs_el[3]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

        #boundary 3-0 : x,y dofs
        idof0=icon[3,iel]
        idof1=icon[0,iel]
        if (bc_fixT[idof0] and bc_fixT[idof1]):  
             idofTr0=bc_nbT[idof0]   
             idofTr1=bc_nbT[idof1]
             rhs_cbf[idofTr0]+=rhs_el[3]   
             rhs_cbf[idofTr1]+=rhs_el[0]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]



    print("     -> M_prime (m,M) %.4e %.4e " %(np.min(M_prime),np.max(M_prime)))
    print("     -> rhs_cbf (m,M) %.4e %.4e " %(np.min(rhs_cbf),np.max(rhs_cbf)))

    sol=sps.linalg.spsolve(sps.csr_matrix(M_prime),rhs_cbf)#,use_umfpack=True)


    for i in range(nnp):
        if bc_fixT[i]:
            dTdy_CBF[i]=sol[bc_nbT[i]]


    Nusselt_CBF=0
    for i in range(nnp-nnx,nnp):
        Nusselt_CBF += dTdy_CBF[i]*hx

    print("CBF_Nusselt = ",Nusselt_CBF)

    if (istep%20 == 0):
        fig,ax=plt.subplots()
        ax.plot(dTdy_CBF[nnp-nnx:nnp])
        fig.savefig("top_flux_"+str(istep)+".pdf")


    ################################################################
    ### Check Convergence ####
    ################################################################


    if istep != 0:
        chi_Nu = np.abs(Nusselt[istep]-Nusselt[istep-1])/Nusselt[istep]
        chi_vrms = np.abs(vrms[istep]-vrms[istep-1])/vrms[istep]

        print("chi_Nu = ", chi_Nu)
        print("chi_vrms = ", chi_vrms)

        if (chi_Nu < tol) and (chi_vrms < tol) :
            print("CONVERGED")
            break






#==============================================================================
# end time stepping loop
#==============================================================================


#####################################################################
# retrieve pressure
#####################################################################

print("-----------------------------")
print("compure pressure and field derivatives")

p=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
qx=np.zeros(nel,dtype=np.float64)  
qy=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
dens=np.zeros(nel,dtype=np.float64)  

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

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    Tc=0.
    for k in range(0,m):
        Tc += N[k]*T[icon[k,iel]]
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        qx[iel]+=-hcond*dNdx[k]*T[icon[k,iel]]
        qy[iel]+=-hcond*dNdy[k]*T[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    p[iel]=-penalty*(exx[iel]+eyy[iel])
    dens[iel]=rho(rho0,alpha,Tc,T0)






print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("dens (m,M) %.4f %.4f " %(np.min(dens),np.max(dens)))
print("qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
print("qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
np.savetxt('temperature.ascii',np.array([x,y,T]).T,header='# x,y,T')
np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
np.savetxt('heatflux.ascii',np.array([xc,yc,qx,qy]).T,header='# xc,yc,qx,qy')
np.savetxt('vrms.ascii',np.array([time,vrms]).T,header='# time, vrms')
np.savetxt('Nusselt.ascii',np.array([time,Nusselt]).T,header='# time, Nu')

#####################################################################
# plot of solution
#####################################################################

u_temp=np.reshape(u,(nnx,nny))
v_temp=np.reshape(v,(nnx,nny))
T_temp=np.reshape(T,(nnx,nny))
p_temp=np.reshape(p,(nelx,nely))
exx_temp=np.reshape(exx,(nelx,nely))
eyy_temp=np.reshape(eyy,(nelx,nely))
exy_temp=np.reshape(exy,(nelx,nely))
dens_temp=np.reshape(dens,(nelx,nely))
qx_temp=np.reshape(qx,(nelx,nely))
qy_temp=np.reshape(qy,(nelx,nely))

fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(18,18))

uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

im = axes[0][0].imshow(u_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][0].set_title('$v_x$', fontsize=10, y=1.01)
axes[0][0].set_xlabel('x')
axes[0][0].set_ylabel('y')
fig.colorbar(im,ax=axes[0][0])

im = axes[0][1].imshow(v_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][1].set_title('$v_y$', fontsize=10, y=1.01)
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
fig.colorbar(im,ax=axes[0][1])

im = axes[0][2].imshow(p_temp,extent=pextent,cmap='RdGy',interpolation='nearest')
axes[0][2].set_title('$p$', fontsize=10, y=1.01)
axes[0][2].set_xlim(0,Lx)
axes[0][2].set_ylim(0,Ly)
axes[0][2].set_xlabel('x')
axes[0][2].set_ylabel('y')
fig.colorbar(im,ax=axes[0][2])

im = axes[0][3].imshow(T_temp,extent=uextent,cmap='jet',interpolation='nearest')
axes[0][3].set_title('$T$', fontsize=10, y=1.01)
axes[0][3].set_xlabel('x')
axes[0][3].set_ylabel('y')
fig.colorbar(im,ax=axes[0][3])

im = axes[1][0].imshow(exx_temp,extent=pextent, cmap='viridis',interpolation='nearest')
axes[1][0].set_title('$\dot{\epsilon}_{xx}$',fontsize=10, y=1.01)
axes[1][0].set_xlim(0,Lx)
axes[1][0].set_ylim(0,Ly)
axes[1][0].set_xlabel('x')
axes[1][0].set_ylabel('y')
fig.colorbar(im,ax=axes[1][0])

im = axes[1][1].imshow(eyy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][1].set_title('$\dot{\epsilon}_{yy}$',fontsize=10,y=1.01)
axes[1][1].set_xlim(0,Lx)
axes[1][1].set_ylim(0,Ly)
axes[1][1].set_xlabel('x')
axes[1][1].set_ylabel('y')
fig.colorbar(im,ax=axes[1][1])

im = axes[1][2].imshow(exy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][2].set_title('$\dot{\epsilon}_{xy}$',fontsize=10,y=1.01)
axes[1][2].set_xlim(0,Lx)
axes[1][2].set_ylim(0,Ly)
axes[1][2].set_xlabel('x')
axes[1][2].set_ylabel('y')
fig.colorbar(im,ax=axes[1][2])

im = axes[1][3].imshow(dens_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[1][3].set_title('$rho$', fontsize=10, y=1.01)
axes[1][3].set_xlabel('x')
axes[1][3].set_ylabel('y')
fig.colorbar(im,ax=axes[1][3])

im = axes[2][0].imshow(qx_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[2][0].set_title('$q_x$', fontsize=10, y=1.01)
axes[2][0].set_xlim(0,Lx)
axes[2][0].set_ylim(0,Ly)
axes[2][0].set_xlabel('x')
axes[2][0].set_ylabel('y')
fig.colorbar(im,ax=axes[2][0])

im = axes[2][1].imshow(qy_temp,extent=uextent,cmap='RdYlBu',interpolation='nearest')
axes[2][1].set_title('$q_y$', fontsize=10, y=1.01)
axes[2][1].set_xlim(0,Lx)
axes[2][1].set_ylim(0,Ly)
axes[2][1].set_xlabel('x')
axes[2][1].set_ylabel('y')
fig.colorbar(im,ax=axes[2][1])

im = axes[2][2].plot(time,vrms)
axes[2][2].set_xlabel('time')
axes[2][2].set_ylabel('$v_{rms}$')

im = axes[2][3].plot(time,Nusselt)
axes[2][3].set_xlabel('time')
axes[2][3].set_ylabel('$Nu$')


plt.subplots_adjust(hspace=0.5)

plt.savefig('solution_convection_box.pdf', bbox_inches='tight')
plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
