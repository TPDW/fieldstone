import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
import tkinter
from scipy import stats


#------------------------------------------------------------------------------

def velocity_x(x,y,ibench):
    if ibench == 0:
        return 0
    elif ibench == 1:
        return 0 
    elif ibench == 2:
        return 0
    elif ibench == 3:
        return u_0*(1-y**2)
    else:
        print("no other benchmarks yet implemented")

def velocity_y(x,y,ibench):
    if ibench == 0:
        return (+2*y-2*y**3-6*x**2*y)/(-3*x**2*y**2+x**2)
    elif ibench == 1:
        return 0
    elif ibench == 2:
        return 0
    elif ibench == 3:
        return 0
    else:
        print("no other benchmarks yet implemented")


def heating(x,y,ibench):
    if ibench == 0  or ibench==1 or ibench==3:
        return 0
    elif ibench==2:
        return 1
    else:
        print("no other benchmarks yet implemented")

def temp_analytical(x,y,ibench):
    if ibench==0:
        return y*x**2-y**3*x**2
    elif ibench==1:
        return y
    elif ibench==2:
        return 1.5*y-0.5*y**2
    elif ibench==3:
        return 1/3*(1-y**4)*u_0**2/hcond*eta
    else:
        print("no other benchmarks yet implemented")


def analytical_shear_heating(x,y,ibench):
    if ibench == 3:
        return 4*eta*y**2*u_0**2
    else:
        return 0



def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(x,y)[0]



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

ibench=3

if ibench==0:
    Lx=1.        # horizontal extent of the domain 
    Ly=0.25       # vertical extent of the domain    
    offset_x=3.0
    offset_y=0.75

elif ibench==1 or ibench==2:
    Lx=1
    Ly=1   
    offset_x=0
    offset_y=0

elif ibench==3:
    Lx=2
    Ly=2
    offset_x=-1
    offset_y=-1



hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density

eta=1        # viscocity

u_0=1

use_analytical_shear_heating=False

L1_temp_list=[]
L2_temp_list=[]

nelx_list = [16,24,32,40,48,56,64]

for nelx in nelx_list:

    print("nelx = %3i" % nelx)
    nely=nelx

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
            x[counter]=i*hx+offset_x
            y[counter]=j*hy+offset_y
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
    # define temperature boundary conditions
    #####################################################################

    print("defining temperature boundary conditions")

    bc_fixT=np.zeros(NfemT,dtype=np.bool)  
    bc_valT=np.zeros(NfemT,dtype=np.float64) 


    for i in range(0,nnp):
        if y[i]>(Ly+offset_y-eps):
            if ibench==0:
                bc_fixT[i]=True ; bc_valT[i]=0.
            elif ibench==1 or ibench ==2:
                bc_fixT[i]=True ; bc_valT[i]=1
            elif ibench==3:
                bc_fixT[i]=True ; bc_valT[i]=0

        if y[i]<(offset_y+eps):
            if ibench==0:
                bc_fixT[i]=True ; bc_valT[i]=y[i]*x[i]**2-y[i]**3*x[i]**2
            if ibench==1 or ibench ==2:
                bc_fixT[i]=True ; bc_valT[i]=0
            elif ibench==3:
                bc_fixT[i]=True ; bc_valT[i]=0


        if x[i]<(offset_x+eps):
            if ibench==0:
                bc_fixT[i]=True ; bc_valT[i]=y[i]*x[i]**2-y[i]**3*x[i]**2

        if x[i]>(Lx+offset_x-eps):
            if ibench==0:
                bc_fixT[i]=True ; bc_valT[i]=y[i]*x[i]**2-y[i]**3*x[i]**2


    NfemTCBF=sum(bc_fixT)

    bc_nbT=np.zeros(NfemT,dtype=np.int32)  # boundary condition, yes/no

    counter=0
    for i in range(0,NfemT):
      if (bc_fixT[i]):
         bc_nbT[i]=counter
         counter+=1


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
    Nusselt=0
    dt=1
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    vel=np.zeros((1,ndim),dtype=np.float64)
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    Nusselt_el=np.zeros(nelx)


    #####################################################################
    # initial velocity
    #####################################################################

    T = np.zeros(nnp,dtype=np.float64)
    t_prev = np.zeros(nnp,dtype=np.float64)
    for i in range(0,nnp):
        u[i]=velocity_x(x[i],y[i],ibench)
        v[i]=velocity_y(x[i],y[i],ibench)


    ######################################################################
    # compute strainrate 
    ######################################################################

    xc = np.zeros(nel,dtype=np.float64)  
    yc = np.zeros(nel,dtype=np.float64)  
    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        rq = 0.0
        sq = 0.0

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


    ######################################################################
    # compute nodal pressure & strainrates : C->N method
    ######################################################################

    exxn1=np.zeros(nnp,dtype=np.float64)  
    eyyn1=np.zeros(nnp,dtype=np.float64)  
    exyn1=np.zeros(nnp,dtype=np.float64)  
    count=np.zeros(nnp,dtype=np.float64)  

    for iel in range(0,nel):
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

    exxn1/=count
    eyyn1/=count
    exyn1/=count


    ###################################################################
    # frictional heating/viscous dissipation
    ###################################################################
    phi_nodal=np.zeros(nnp,dtype=np.float64)

    for i in range(nnp):
        phi_nodal[i] = eta*(2*exxn1[i]**2 + 2*eyyn1[i]**2 +(2*exyn1[i])**2)

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
                xq=0
                yq=0
                phiq=0
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                    xq+=x[icon[k,iel]]*N_mat[k,0]
                    yq+=y[icon[k,iel]]*N_mat[k,0]
                    phiq+=phi_nodal[icon[k,iel]]*N_mat[k,0]

                # compute mass matrix
                #MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                a_el=(Ka+Kd)#*dt +MM

                if use_analytical_shear_heating:
                    b_el=N_mat*(heating(xq,yq,ibench)+analytical_shear_heating(xq,yq,ibench))*wq*jcob
                else:
                    if ibench==3:
                        b_el=N_mat*(heating(xq,yq,ibench)+phiq)*wq*jcob
                    else:
                        b_el=N_mat*heating(xq,yq,ibench)*wq*jcob



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

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve T time: %.3f s" % (timing.time() - start))

    print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))


    #################################################################
    # compute Nusselt number at top
    #################################################################

    Nusselt_el=np.zeros(nelx)
    counter=0
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
        if y[icon[3,iel]]>(Ly+offset_y-eps):
           Nusselt+=qy*hx
           Nusselt_el[counter]=qy
           counter+=1

    print(" Nusselt= %.6f" %(Nusselt))
    #print(" Nusselt (theoretical)=%.6f" %(Nusselt_analytical))



    #####################################
    # Temp Convergence
    #####################################
    L1_temp=0
    L2_temp=0

    # for i in range(nnp):
    #     L1_temp +=abs(T[i]-temp_analytical(x[i],y[i],ibench))/nnp
    #     L2_temp +=(T[i]-temp_analytical(x[i],y[i],ibench))**2/nnp

    for iel in range(nel):
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


                xq=0.0
                yq=0.0
                Tq=0.0
                for k in range(0, m):
                    xq+=N_mat[k,0]*x[icon[k,iel]]
                    #print(N[k],x[icon[k,iel]])
                    yq+=N_mat[k,0]*y[icon[k,iel]]
                    Tq+=N_mat[k,0]*T[icon[k,iel]]


                L1_temp+=abs(Tq-temp_analytical(xq,yq,ibench))*wq*jcob
                #print(Tq,temp_analytical(xq,yq,ibench),xq,yq)

                L2_temp+=(Tq-temp_analytical(xq,yq,ibench))**2*wq*jcob




    L1_temp_list.append(L1_temp)
    L2_temp_list.append(np.sqrt(L2_temp))

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
# vtufile.write("<CellData Scalars='scalars'>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='residual' Format='ascii'> \n")
# for iel in range (0,nel):
#     vtufile.write("%10e\n" % rhs_cbf_domain[iel])
# vtufile.write("</DataArray>\n")

# vtufile.write("</CellData>\n")


#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%20e \n" %T[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='x' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%20e \n" %x[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='y' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%20e \n" %y[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='boundaries' Format='ascii'> \n")
for i in range(0,nnp):
    if bc_fixT[i]:
        vtufile.write("%20e \n" %1)
    else:
        vtufile.write("%20e \n" %0)
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='boundary conditions' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%20e \n" %bc_valT[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='temperature errors' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%20e \n" %(T[i]-temp_analytical(x[i],y[i],ibench)))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='temperature errors fractional' Format='ascii'> \n")
for i in range(0,nnp):
    value=((T[i]-temp_analytical(x[i],y[i],ibench))/(temp_analytical(x[i],y[i],ibench)+eps))
    vtufile.write("%20e \n" %value)
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical phi' Format='ascii'> \n")
for i in range(0,nnp):
    value=analytical_shear_heating(x[i],y[i],ibench)
    vtufile.write("%20e \n" %value)
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='calculated phi' Format='ascii'> \n")
for i in range(0,nnp):
    value=phi_nodal[i]
    vtufile.write("%20e \n" %value)
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




hx_list = np.log(1/(np.array(nelx_list)+1))



plt.plot(hx_list,np.log(np.array(L1_temp_list)),label="L1")
plt.plot(hx_list,np.log(np.array(L2_temp_list)),label="L2")
plt.xlabel("log(hx)")
plt.ylabel("Norm Magnitude")
plt.legend()
plt.savefig("temp_conv"+str(ibench)+".pdf")

print ("The L1 temperature norm slope is %10f" % get_regression(hx_list,np.log(np.array(L1_temp_list))))
print ("The L2 temperature norm slope is %10f" % get_regression(hx_list,np.log(np.array(L2_temp_list))))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
