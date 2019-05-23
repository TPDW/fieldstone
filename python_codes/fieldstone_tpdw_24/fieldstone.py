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

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# # allowing for argument parsing through command line
# if int(len(sys.argv) == 4):
#    nelx = int(sys.argv[1])
#    nely = int(sys.argv[2])
#    visu = int(sys.argv[3])
# else:
#    nelx = 24
#    nely = 12
#    visu = 1

# assert (nelx>0.), "nnx should be positive" 
# assert (nely>0.), "nny should be positive" 


nelx_list = [4,8,10,12,14]
#nelx_list=[64,64]

L1_norm_list_CBF=[[] for Null in range(8)]
L2_norm_list_CBF=[[] for Null in range(8)]

L1_norm_list_CN=[[] for Null in range(8)]
L2_norm_list_CN=[[] for Null in range(8)]

L1_norm_list_LS=[[] for Null in range(8)]
L2_norm_list_LS=[[] for Null in range(8)]



fig_CN,axes_CN=plt.subplots(1,len(nelx_list),figsize=(30,10))

counter_nelx=0

for nelx in nelx_list:
    print("----------------------")
    print("Now Running nelx=" + str(nelx))
    print("----------------------")

    nely=nelx     

    
    nnx=2*nelx+1  # number of elements, x direction
    nny=2*nely+1  # number of elements, y direction

    nnp=nnx*nny  # number of nodes

    nel=nelx*nely  # number of elements, total

    viscosity=1.  # dynamic viscosity \mu

    NfemV=nnp*ndofV               # number of velocity dofs
    NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
    Nfem=NfemV+NfemP              # total number of dofs

    eps=1.e-10
    qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
    qweights=[5./9.,8./9.,5./9.]

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
            x[counter]=i*hx/2.
            y[counter]=j*hy/2.
            counter += 1

    np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

    print("setup: grid points: %.3f s" % (time.time() - start))

    #################################################################
    # connectivity
    #################################################################
    # velocity    pressure
    # 3---6---2   3-------2
    # |       |   |       |
    # 7   8   5   |       |
    # |       |   |       |
    # 0---4---1   0-------1
    #################################################################


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

    #for iel in range (0,nel):
    #    print ("iel=",iel)
    #    print ("node 1",iconV[0][iel],"at pos.",x[iconV[0][iel]], y[iconV[0][iel]])
    #    print ("node 2",iconV[1][iel],"at pos.",x[iconV[1][iel]], y[iconV[1][iel]])
    #    print ("node 3",iconV[2][iel],"at pos.",x[iconV[2][iel]], y[iconV[2][iel]])
    #    print ("node 4",iconV[3][iel],"at pos.",x[iconV[3][iel]], y[iconV[3][iel]])
    #    print ("node 2",iconV[4][iel],"at pos.",x[iconV[4][iel]], y[iconV[4][iel]])
    #    print ("node 2",iconV[5][iel],"at pos.",x[iconV[5][iel]], y[iconV[5][iel]])
    #    print ("node 2",iconV[6][iel],"at pos.",x[iconV[6][iel]], y[iconV[6][iel]])
    #    print ("node 2",iconV[7][iel],"at pos.",x[iconV[7][iel]], y[iconV[7][iel]])
    #    print ("node 2",iconV[8][iel],"at pos.",x[iconV[8][iel]], y[iconV[8][iel]])

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
    #     print ("node 1",iconV[0][iel],"at pos.",x[iconV[0][iel]], y[iconV[0][iel]])
    #     print ("node 2",iconV[1][iel],"at pos.",x[iconV[1][iel]], y[iconV[1][iel]])
    #     print ("node 3",iconV[2][iel],"at pos.",x[iconV[2][iel]], y[iconV[2][iel]])
    #     print ("node 4",iconV[3][iel],"at pos.",x[iconV[3][iel]], y[iconV[3][iel]])

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
           bc_nb[i]=counter
           counter+=1

    print (np.min(bc_nb))
    print (np.max(bc_nb))



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

        rq = 0.0
        sq = 0.0
        weightq = 2.0 * 2.0

        NV[0:9]=NNV(rq,sq)
        dNVdr[0:9]=dNNVdr(rq,sq)
        dNVds[0:9]=dNNVds(rq,sq)

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
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NV[0:9]=NNV(rq,sq)
                dNVdr[0:9]=dNNVdr(rq,sq)
                dNVds[0:9]=dNNVds(rq,sq)
                NP[0:4]=NNP(rq,sq)

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

    print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

    print("compute errors: %.3f s" % (time.time() - start))

    #####################################################################
    # interpolate pressure onto velocity grid points
    #####################################################################

    q=np.zeros(nnp,dtype=np.float64)

    for iel in range(0,nel):
        q[iconV[0,iel]]=p[iconP[0,iel]]
        q[iconV[1,iel]]=p[iconP[1,iel]]
        q[iconV[2,iel]]=p[iconP[2,iel]]
        q[iconV[3,iel]]=p[iconP[3,iel]]
        q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
        q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
        q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
        q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
        q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

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

    # M_prime_el =(hx/2.)*np.array([ \
    # [2./3.,1./3.],\
    # [1./3.,2./3.]])

    M_prime_el=np.array([\
    [ 4/15, 2/15,-1/15],\
    [ 2/15,16/15, 2/15],\
    [-1/15, 2/15, 4/15]])


    #Mass Lumping
    # M_prime_el=(hx/2)*np.array([\
    # [1,0,0],\
    # [0,1,0],\
    # [0,0,1]])


    M_prime_el=np.zeros((3,3),dtype=np.float64)
    use_mass_lumping=True
    if use_mass_lumping:
        degree=3
        gleg_points,gleg_weights=gauss_lobatto(degree,20)
    else:
        degree=3
        gleg_points,gleg_weights=np.polynomial.legendre.leggauss(degree)
    for iq in range(degree):
        jq=0
        rq=gleg_points[iq]
        sq=-1
        weightq=gleg_weights[iq]
        
        NV[0:mV]=NNV(rq,sq)
    #     dNdr[0:m]=dNNVdr(rq,sq)
    #     dNds[0:m]=dNNVds(rq,sq)

    #     # calculate jacobian matrix
    #     jcb = np.zeros((2, 2),dtype=float)
    #     for k in range(0,m):
    #         jcb[0, 0] += dNdr[k]*x[iconV[k,iel]]
    #         jcb[0, 1] += dNdr[k]*y[iconV[k,iel]]
    #         jcb[1, 0] += dNds[k]*x[iconV[k,iel]]
    #         jcb[1, 1] += dNds[k]*y[iconV[k,iel]]
    #     jcob = np.linalg.det(jcb)
    #     jcbi = np.linalg.inv(jcb)  
        

        M_prime_el[0,0] += NV[0]*NV[0]*weightq
        M_prime_el[0,1] += NV[0]*NV[4]*weightq
        M_prime_el[0,2] += NV[0]*NV[1]*weightq

        M_prime_el[1,0] += NV[4]*NV[0]*weightq
        M_prime_el[1,1] += NV[4]*NV[4]*weightq
        M_prime_el[1,2] += NV[4]*NV[1]*weightq

        M_prime_el[2,0] += NV[1]*NV[0]*weightq
        M_prime_el[2,1] += NV[1]*NV[4]*weightq
        M_prime_el[2,2] += NV[1]*NV[1]*weightq



    for iel in range(0,nel):

        #-----------------------
        # compute Kel, Gel, f
        #-----------------------

        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        rhs_el =np.zeros((mV*ndofV),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                # calculate shape functions
                NV[0:mV]=NNV(rq,sq)
                dNVdr[0:mV]=dNNVdr(rq,sq)
                dNVds[0:mV]=dNNVds(rq,sq)
                NP[0:4]=NNP(rq,sq)


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
                    #rhoqV+=N[k]*rho[iconV[k,iel]]
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
                         u[iconV[8,iel]],v[iconV[8,iel]] ])

        # v_el = np.array([u[iconV[0,iel]],v[iconV[0,iel]],\
        #                  u[iconV[1,iel]],v[iconV[1,iel]],\
        #                  u[iconV[2,iel]],v[iconV[2,iel]],\
        #                  u[iconV[3,iel]],v[iconV[3,iel]] ])

        #print(f_el)

        p_el = np.array([p[iconP[0,iel]],\
                         p[iconP[1,iel]],\
                         p[iconP[2,iel]],\
                         p[iconP[3,iel]]])


        rhs_el=-f_el+K_el.dot(v_el)+G_el.dot(p_el)

        #-----------------------
        # assemble 
        #-----------------------

        #boundary 0-1 : x,y dofs
        for i in range(0,ndofV):
            idof0=2*iconV[0,iel]+i
            idof1=2*iconV[4,iel]+i
            idof2=2*iconV[1,iel]+i
            if (bc_fix[idof0] and bc_fix[idof1]):  
               idofTr0=bc_nb[idof0]   
               idofTr1=bc_nb[idof1]
               idofTr2=bc_nb[idof2]
               rhs_cbf[idofTr0]+=rhs_el[0+i]   
               rhs_cbf[idofTr1]+=rhs_el[8+i]   
               rhs_cbf[idofTr2]+=rhs_el[2+i]   
               M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hx/2
               M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hx/2
               M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hx/2
               M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hx/2
               M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hx/2
               M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hx/2
               M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hx/2
               M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hx/2
               M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hx/2



        #boundary 1-2 : x,y dofs
        for i in range(0,ndofV):
            idof0=2*iconV[1,iel]+i
            idof1=2*iconV[5,iel]+i
            idof2=2*iconV[2,iel]+i
            if (bc_fix[idof0] and bc_fix[idof1]):  
               idofTr0=bc_nb[idof0]   
               idofTr1=bc_nb[idof1]
               idofTr2=bc_nb[idof2]
               rhs_cbf[idofTr0]+=rhs_el[2+i]   
               rhs_cbf[idofTr1]+=rhs_el[10+i]   
               rhs_cbf[idofTr2]+=rhs_el[4+i]   
               M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hy/2
               M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hy/2
               M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hy/2
               M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hy/2
               M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hy/2
               M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hy/2
               M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hy/2
               M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hy/2
               M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hy/2


        #boundary 2-3 : x,y dofs
        for i in range(0,ndofV):
            idof0=2*iconV[2,iel]+i
            idof1=2*iconV[6,iel]+i
            idof2=2*iconV[3,iel]+i
            if (bc_fix[idof0] and bc_fix[idof1]):  
               idofTr0=bc_nb[idof0]   
               idofTr1=bc_nb[idof1]
               idofTr2=bc_nb[idof2]
               rhs_cbf[idofTr0]+=rhs_el[4+i]   
               rhs_cbf[idofTr1]+=rhs_el[12+i]   
               rhs_cbf[idofTr2]+=rhs_el[6+i]   
               M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hx/2
               M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hx/2
               M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hx/2
               M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hx/2
               M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hx/2
               M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hx/2
               M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hx/2
               M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hx/2
               M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hx/2


        #boundary 3-0 : x,y dofs
        for i in range(0,ndofV):
            idof0=2*iconV[3,iel]+i
            idof1=2*iconV[7,iel]+i
            idof2=2*iconV[0,iel]+i
            if (bc_fix[idof0] and bc_fix[idof1]):  
               idofTr0=bc_nb[idof0]   
               idofTr1=bc_nb[idof1]
               idofTr2=bc_nb[idof2]
               rhs_cbf[idofTr0]+=rhs_el[6+i]   
               rhs_cbf[idofTr1]+=rhs_el[14+i]   
               rhs_cbf[idofTr2]+=rhs_el[0+i]   
               M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*hy/2
               M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*hy/2
               M_prime[idofTr0,idofTr2]+=M_prime_el[0,2]*hy/2
               M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*hy/2
               M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*hy/2
               M_prime[idofTr1,idofTr2]+=M_prime_el[1,2]*hy/2
               M_prime[idofTr2,idofTr0]+=M_prime_el[2,0]*hy/2
               M_prime[idofTr2,idofTr1]+=M_prime_el[2,1]*hy/2
               M_prime[idofTr2,idofTr2]+=M_prime_el[2,2]*hy/2



        # #boundary 1-2 : x,y dofs
        # for i in range(0,ndofV):
        #     idof0=2*iconV[1,iel]+i
        #     idof1=2*iconV[2,iel]+i
        #     if (bc_fix[idof0] and bc_fix[idof1]):  
        #        idofTr0=bc_nb[idof0]   
        #        idofTr1=bc_nb[idof1]
        #        rhs_cbf[idofTr0]+=rhs_el[2+i]   
        #        rhs_cbf[idofTr1]+=rhs_el[4+i]   
        #        M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
        #        M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
        #        M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
        #        M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

        # #boundary 2-3 : x,y dofs
        # for i in range(0,ndofV):
        #     idof0=2*iconV[2,iel]+i
        #     idof1=2*iconV[3,iel]+i
        #     if (bc_fix[idof0] and bc_fix[idof1]):  
        #        idofTr0=bc_nb[idof0]   
        #        idofTr1=bc_nb[idof1]
        #        rhs_cbf[idofTr0]+=rhs_el[4+i]   
        #        rhs_cbf[idofTr1]+=rhs_el[6+i]   
        #        M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
        #        M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
        #        M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
        #        M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

        # #boundary 3-0 : x,y dofs
        # for i in range(0,ndofV):
        #     idof0=2*iconV[3,iel]+i
        #     idof1=2*iconV[0,iel]+i
        #     if (bc_fix[idof0] and bc_fix[idof1]):  
        #        idofTr0=bc_nb[idof0]   
        #        idofTr1=bc_nb[idof1]
        #        rhs_cbf[idofTr0]+=rhs_el[6+i]   
        #        rhs_cbf[idofTr1]+=rhs_el[0+i]   
        #        M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]
        #        M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]
        #        M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]
        #        M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]

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


    ############################################################################
    ######## Calculate the L1 and L2 norms for ty on the all surfaces ##########
    ############################################################################

    left_elements_list=[]
    right_elements_list=[]
    for iel in range(nel):
        if (x[iconV[0,iel]] < eps):
          left_elements_list.append(iel)
        if (abs(x[iconV[1,iel]]-Lx) < eps):
          right_elements_list.append(iel)


    L1_norm_CBF=[0]*8
    L2_norm_CBF=[0]*8
    L1_norm_CN =[0]*8
    L2_norm_CN =[0]*8
    L1_norm_LS =[0]*8
    L2_norm_LS =[0]*8

    degree=4
    gleg_points,gleg_weights=np.polynomial.legendre.leggauss(degree)


    for iel in range(nel-nelx+1,nel-1): # upper norms 
        for iq in range(degree):
            rq=gleg_points[iq]
            weightq=gleg_weights[iq]

            N_0=0.5*rq*(1-rq)
            N_1=0.5*rq*(1+rq)
            N_2=1-rq**2

            xq=N_0*x[iconV[3,iel]]+N_1*x[iconV[2,iel]]+N_2*x[iconV[6,iel]]

            tyq_CBF=N_0*ty[iconV[3,iel]]+N_1*ty[iconV[2,iel]]+N_2*ty[iconV[6,iel]]
            txq_CBF=N_0*tx[iconV[3,iel]]+N_1*tx[iconV[2,iel]]+N_2*tx[iconV[6,iel]]

            L1_norm_CBF[0]+=abs(tyq_CBF-sigma_yy(xq,1))*hx*weightq
            L2_norm_CBF[0]+=(tyq_CBF-sigma_yy(xq,1))**2*hx*weightq

            L1_norm_CBF[1]+=abs(txq_CBF-sigma_xy(xq,1))*hx*weightq
            L2_norm_CBF[1]+=(txq_CBF-sigma_xy(xq,1))**2*hx*weightq




    for iel in range(1,nelx-1): # lower norms 
        for iq in range(degree):
            rq=gleg_points[iq]
            weightq=gleg_weights[iq]

            N_0=0.5*rq*(1-rq)
            N_1=0.5*rq*(1+rq)
            N_2=1-rq**2

            xq=N_0*x[iconV[0,iel]]+N_1*x[iconV[1,iel]]+N_2*x[iconV[4,iel]]


            tyq_CBF=N_0*ty[iconV[0,iel]]+N_1*ty[iconV[1,iel]]+N_2*ty[iconV[4,iel]]
            txq_CBF=N_0*tx[iconV[0,iel]]+N_1*tx[iconV[1,iel]]+N_2*tx[iconV[4,iel]]

            L1_norm_CBF[2]+=abs(tyq_CBF-sigma_yy(xq,0))*hx*weightq
            L2_norm_CBF[2]+=(tyq_CBF-sigma_yy(xq,0))**2*hx*weightq

            L1_norm_CBF[3]+=abs(txq_CBF-sigma_xy(xq,0))*hx*weightq
            L2_norm_CBF[3]+=(txq_CBF-sigma_xy(xq,0))**2*hx*weightq

    for i in range(1,len(left_elements_list)-1): #left norms ergo nodes 3 & 0
        iel=left_elements_list[i]
        for iq in range(degree):
            rq=gleg_points[iq]
            weightq=gleg_weights[iq]

            N_0=0.5*rq*(1-rq)
            N_1=0.5*rq*(1+rq)
            N_2=1-rq**2

            yq=N_0*x[iconV[1,iel]]+N_1*x[iconV[2,iel]]+N_2*x[iconV[5,iel]]


            tyq_CBF=N_0*ty[iconV[1,iel]]+N_1*ty[iconV[2,iel]]+N_2*ty[iconV[5,iel]]
            txq_CBF=N_0*tx[iconV[1,iel]]+N_1*tx[iconV[2,iel]]+N_2*tx[iconV[5,iel]]

            L1_norm_CBF[4]+=abs(tyq_CBF-sigma_xy(0,yq))*hx*weightq
            L2_norm_CBF[4]+=(tyq_CBF-sigma_xy(0,yq))**2*hx*weightq

            L1_norm_CBF[5]+=abs(txq_CBF-sigma_xx(0,yq))*hx*weightq
            L2_norm_CBF[5]+=(txq_CBF-sigma_xx(0,yq))**2*hx*weightq

    for i in range(1,len(right_elements_list)-1): #left norms ergo nodes 3 & 0
        iel=right_elements_list[i]
        for iq in range(degree):
            rq=gleg_points[iq]
            weightq=gleg_weights[iq]

            N_0=0.5*rq*(1-rq)
            N_1=0.5*rq*(1+rq)
            N_2=1-rq**2

            yq=N_0*x[iconV[0,iel]]+N_1*x[iconV[3,iel]]+N_2*x[iconV[7,iel]]


            tyq_CBF=N_0*ty[iconV[0,iel]]+N_1*ty[iconV[3,iel]]+N_2*ty[iconV[7,iel]]
            txq_CBF=N_0*tx[iconV[0,iel]]+N_1*tx[iconV[3,iel]]+N_2*tx[iconV[7,iel]]

            L1_norm_CBF[6]+=abs(tyq_CBF-sigma_xy(0,yq))*hx*weightq
            L2_norm_CBF[6]+=(tyq_CBF-sigma_xy(0,yq))**2*hx*weightq

            L1_norm_CBF[7]+=abs(txq_CBF-sigma_xx(0,yq))*hx*weightq
            L2_norm_CBF[7]+=(txq_CBF-sigma_xx(0,yq))**2*hx*weightq


    for i in range(8):
        print(i)
        L1_norm_list_CBF[i].append(L1_norm_CBF[i])
        L2_norm_list_CBF[i].append(np.sqrt(L2_norm_CBF[i]))



hx_list=1/(2*np.array(nelx_list)+1)
log_hx=np.log(np.abs(hx_list))

fig,ax=plt.subplots(4,2,figsize=(20,40))

for i in range(8):
  print(i)
  print(i%2)
  print(int(np.floor(i/2)))
  #ax.plot(log_hx,np.log(np.abs(corner_error_list)),label="corner CBF")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_CBF[i]),label="L1 CBF")
  ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_CBF[i]),label="L2 CBF")
  # ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_CN[i]),label="L1 CN")
  # ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_CN[i]),label="L2 CN")
  # ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L1_norm_list_LS[i]),label="L1 LS")
  # ax[int(np.floor(i/2))][i%2].plot(log_hx,np.log(L2_norm_list_LS[i]),label="L2 LS")
  ax[int(np.floor(i/2))][i%2].grid()
  ax[int(np.floor(i/2))][i%2].legend()
  ax[int(np.floor(i/2))][i%2].set_xlabel("log(1/nnx)"+ str(i))
  ax[int(np.floor(i/2))][i%2].set_ylabel("Norm Magnitude")
  #ax[i%2][np.floor(i/2)].set_title("Convergence Rates for Traction Norms")
fig.savefig("convergence.pdf")


#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

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
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
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
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %23)
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
