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

def rho(rho0,alpha,T,T0):
    val=rho0*(1.-alpha*(T-T0))
    return val

def mu(T):
    val=1.
    return val


def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(x,y)[0]

def get_regression_x_intercept_difference(h,y1,y2):
  [m1,c1]=stats.linregress(h,y1)[0:2]
  [m2,c2]=stats.linregress(h,y2)[0:2]
  r1=-c1/m1
  r2=-c2/m2

  return r1-r2

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
Ly=0.25       # vertical extent of the domain 

hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density




# # allowing for argument parsing through command line
# if int(len(sys.argv) == 3):
#    nelx = int(sys.argv[1])
#    nely = int(sys.argv[2])
# else:
#    nelx = 64
#    nely = 64

nelx_list = [16,24,32,40,48,56,64,128]

Nusselt_CBF_list=[]
Nusselt_elemental_list=[]

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

    offset_x=3.0
    offset_y=0.75

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
           bc_fixT[i]=True ; bc_valT[i]=0.
           # print(i)
           # print(y[i])

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



    #####################################################################
    # initial temperature and velocity
    #####################################################################

    T = np.zeros(nnp,dtype=np.float64)
    t_prev = np.zeros(nnp,dtype=np.float64)
    for i in range(0,nnp):
        T[i]=y[i]*x[i]**2-y[i]**3*x[i]**2
        u[i]=0
        v[i]=(+2*y[i]-2*y[i]**3-6*x[i]**2*y[i])/(-3*x[i]**2*y[i]**2+x[i]**2)



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
           #Nusselt_el[counter]=qy
           counter+=1

    print(" Nusselt= %.6f" %(Nusselt))
    Nusselt_elemental_list.append(Nusselt)

    ################################################################
    ####### CBF
    ################################################################


    M_prime = np.zeros((NfemTCBF,NfemTCBF),np.float64)
    rhs_cbf = np.zeros(NfemTCBF,np.float64)
    dTdy_CBF = np.zeros(nnp,np.float64)

    rhs_cbf_domain=np.zeros(nnp,np.float64)

    # M_prime_el =(hx/2.)*np.array([ \
    # [2./3.,1./3.],\
    # [1./3.,2./3.]])

    M_prime_el =(hx/2.)*np.array([ \
    [1,0],\
    [0,1]])


    for iel in range(nel):

        T_el = np.array([T[icon[0,iel]],T[icon[1,iel]],T[icon[2,iel]],T[icon[3,iel]]])
        T_prev_el = np.array([t_prev[icon[0,iel]],t_prev[icon[1,iel]],t_prev[icon[2,iel]],t_prev[icon[3,iel]]])

        T_el_dot = (T_el - T_prev_el)*0

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

        for k in range(0,m):
            rhs_cbf_domain[icon[k,iel]]+=rhs_el[k]
      #-----------------------
      # assemble 
      #-----------------------

        # #boundary 0-1 : x,y dofs
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

        # #boundary 1-2 : x,y dofs
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

        # #boundary 3-0 : x,y dofs
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

    Nusselt_CBF_list.append(Nusselt_CBF)



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
# # vtufile.write("<CellData Scalars='scalars'>\n")
# # #--
# # vtufile.write("<DataArray type='Float32' Name='residual' Format='ascii'> \n")
# # for iel in range (0,nel):
# #     vtufile.write("%10e\n" % rhs_cbf_domain[iel])
# # vtufile.write("</DataArray>\n")

# # vtufile.write("</CellData>\n")


# #####
# vtufile.write("<PointData Scalars='scalars'>\n")
# #--
# vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e \n" %T[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='x' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e \n" %x[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='y' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e \n" %y[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='residual' Format='ascii'> \n")
# for i in range(0,nnp):
#     vtufile.write("%20e \n" %rhs_cbf_domain[i])
# vtufile.write("</DataArray>\n")
# #--
# vtufile.write("<DataArray type='Float32' Name='boundaries' Format='ascii'> \n")
# for i in range(0,nnp):
#     if bc_fixT[i]:
#         vtufile.write("%20e \n" %1)
#     else:
#         vtufile.write("%20e \n" %0)
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

Nusselt_analytical=74/3

hx_list = np.log(1/(np.array(nelx_list)+1))
Nusselt_error_cbf_list = np.log(np.abs(np.array(Nusselt_CBF_list)- Nusselt_analytical))
Nusselt_error_ele_list = np.log(np.abs(np.array(Nusselt_elemental_list)- Nusselt_analytical))


plt.plot(hx_list,Nusselt_error_cbf_list,label="CBF")
plt.plot(hx_list,Nusselt_error_ele_list,label="elemental")
plt.legend()
plt.xlabel("log(1/nnx)")
plt.ylabel("log(Nusselt Error)")
plt.savefig("convergence.pdf")

regression = get_regression(hx_list,Nusselt_error_cbf_list)
print("Regression value is %10f" % regression)

x_intercept= get_regression_x_intercept_difference(hx_list,Nusselt_error_cbf_list,Nusselt_error_ele_list)
print("X intercept value is %10f" % x_intercept)

print("This means that using the ele method is equivalent to using a CBF method with a %5f better resolution" % np.exp(np.abs(x_intercept)))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
