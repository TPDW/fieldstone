import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from scipy import stats


#------------------------------------------------------------------------------
def density(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
    gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
    alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
    val=k*math.sin(k*theta)*alephr + rho0 
    return val

def velocity_x(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.cos(theta)-vtheta*math.sin(theta)
    return val
    #return 4*x*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2) - y*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*cos(4*atan2(y, x))/sqrt(x**2 + y**2)

def velocity_y(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.sin(theta)+vtheta*math.cos(theta)
    return val

def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    #return val
    return 4*(-6*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 2/sqrt(x**2 + y**2) + 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan(y/x))/sqrt(x**2 + y**2)

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

sqrt=np.sqrt
log=np.log
atan=np.arctan
sin=np.sin
cos=np.cos
atan2=np.arctan2
def Rational(a,b):
    return a/b

def s_xx(x,y):
    #This is just taken from Sympy
    return -8*x**2*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 2*x*y*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - 32*x*y*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 8*x*(x/sqrt(x**2 + y**2) + 3*x*log(sqrt(x**2 + y**2))/((x**2 + y**2)**Rational(3, 2)*log(2)) - 3*x/((x**2 + y**2)**Rational(3, 2)*log(2)) + x/(x**2 + y**2)**Rational(3, 2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2) - 8*y**2*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - 2*y*(2*x/sqrt(x**2 + y**2) + 3*x/((x**2 + y**2)**Rational(3, 2)*log(2)))*cos(4*atan2(y, x))/sqrt(x**2 + y**2) - 4*(-6*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 2/sqrt(x**2 + y**2) + 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan(y/x))/sqrt(x**2 + y**2) + 8*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2)

def s_yy(x,y):
    return -8*x**2*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - 2*x*y*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 32*x*y*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 2*x*(2*y/sqrt(x**2 + y**2) + 3*y/((x**2 + y**2)**Rational(3, 2)*log(2)))*cos(4*atan2(y, x))/sqrt(x**2 + y**2) - 8*y**2*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 8*y*(y/sqrt(x**2 + y**2) + 3*y*log(sqrt(x**2 + y**2))/((x**2 + y**2)**Rational(3, 2)*log(2)) - 3*y/((x**2 + y**2)**Rational(3, 2)*log(2)) + y/(x**2 + y**2)**Rational(3, 2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2) - 4*(-6*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 2/sqrt(x**2 + y**2) + 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan(y/x))/sqrt(x**2 + y**2) + 8*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2)

def s_xy(x,y):
    return -x**2*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 16*x**2*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + 8*x*y*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - 8*x*y*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*sin(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) + x*(2*x/sqrt(x**2 + y**2) + 3*x/((x**2 + y**2)**Rational(3, 2)*log(2)))*cos(4*atan2(y, x))/sqrt(x**2 + y**2) + 4*x*(y/sqrt(x**2 + y**2) + 3*y*log(sqrt(x**2 + y**2))/((x**2 + y**2)**Rational(3, 2)*log(2)) - 3*y/((x**2 + y**2)**Rational(3, 2)*log(2)) + y/(x**2 + y**2)**Rational(3, 2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2) + y**2*(2*sqrt(x**2 + y**2) - 3/(sqrt(x**2 + y**2)*log(2)))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - 16*y**2*(sqrt(x**2 + y**2) - 3*log(sqrt(x**2 + y**2))/(sqrt(x**2 + y**2)*log(2)) - 1/sqrt(x**2 + y**2))*cos(4*atan2(y, x))/(x**2 + y**2)**Rational(3, 2) - y*(2*y/sqrt(x**2 + y**2) + 3*y/((x**2 + y**2)**Rational(3, 2)*log(2)))*cos(4*atan2(y, x))/sqrt(x**2 + y**2) + 4*y*(x/sqrt(x**2 + y**2) + 3*x*log(sqrt(x**2 + y**2))/((x**2 + y**2)**Rational(3, 2)*log(2)) - 3*x/((x**2 + y**2)**Rational(3, 2)*log(2)) + x/(x**2 + y**2)**Rational(3, 2))*sin(4*atan2(y, x))/sqrt(x**2 + y**2)



def get_analytical_tx(x,y):
    #first, we need to get the normal
    n_x=x/np.sqrt(x**2+y**2)
    n_y=y/np.sqrt(x**2+y**2)


    if (sqrt(x**2+y**2)>1.5):
        return s_xx(x,y)*n_x+s_xy(x,y)*n_y
    else:
        return -s_xx(x,y)*n_x-s_xy(x,y)*n_y

def get_analytical_ty(x,y):
    #first, we need to get the normal
    n_x=x/np.sqrt(x**2+y**2)
    n_y=y/np.sqrt(x**2+y**2)

    if (sqrt(x**2+y**2)>1.5):
        return s_xy(x,y)*n_x+s_yy(x,y)*n_y
    else:
        return -s_xy(x,y)*n_x-s_yy(x,y)*n_y


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
ndof=2  # number of degrees of freedom per node

# if int(len(sys.argv) == 3):
#    nelr = int(sys.argv[1])
#    visu = int(sys.argv[2])
# else:
#    nelr = 24
#    visu = 1

nelr_list=[8,12,16,20]

L1_tx_inner_list=[]
L1_ty_inner_list=[]
L1_tx_outer_list=[]
L1_ty_outer_list=[]

L2_tx_inner_list=[]
L2_ty_inner_list=[]
L2_tx_outer_list=[]
L2_ty_outer_list=[]

L1_tx_inner_listn1=[]
L1_ty_inner_listn1=[]
L1_tx_outer_listn1=[]
L1_ty_outer_listn1=[]

L2_tx_inner_listn1=[]
L2_ty_inner_listn1=[]
L2_tx_outer_listn1=[]
L2_ty_outer_listn1=[]

for nelr in nelr_list:

    R1=1.
    R2=2.

    dr=(R2-R1)/nelr
    nelt=int(2.*math.pi*R2/dr)
    nel=nelr*nelt  # number of elements, total

    nnr=nelr+1
    nnt=nelt
    nnp=nnr*nnt  # number of nodes

    rho0=0.
    kk=4
    g0=1.

    viscosity=1.  # dynamic viscosity \mu
    penalty=1.e7  # penalty coefficient value

    Nfem=nnp*ndof  # Total number of degrees of freedom


    eps=1.e-10

    sqrt3=np.sqrt(3.)

    #################################################################
    # grid point setup
    #################################################################

    print("grid point setup")

    x=np.empty(nnp,dtype=np.float64)  # x coordinates
    y=np.empty(nnp,dtype=np.float64)  # y coordinates
    r=np.empty(nnp,dtype=np.float64)  
    theta=np.empty(nnp,dtype=np.float64) 

    Louter=2.*math.pi*R2
    Lr=R2-R1
    sx = Louter/float(nelt)
    sz = Lr    /float(nelr)

    counter=0
    for j in range(0,nnr):
        for i in range(0,nelt):
            x[counter]=i*sx
            y[counter]=j*sz
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

    print("connectivity")

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

    #################################################################
    # define boundary conditions
    #################################################################
    start = time.time()

    bc_fix = np.zeros(Nfem, dtype=np.bool)  
    bc_val = np.zeros(Nfem, dtype=np.float64) 
    is_inner=np.zeros(nnp, dtype=np.bool)
    is_outer=np.zeros(nnp, dtype=np.bool)

    for i in range(0, nnp):
        if r[i]<R1+eps:
           bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = velocity_x(x[i],y[i],R1,R2,kk,rho0,g0) 
           bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = velocity_y(x[i],y[i],R1,R2,kk,rho0,g0) 
           is_inner[i]=True
        if r[i]>(R2-eps):
           bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = velocity_x(x[i],y[i],R1,R2,kk,rho0,g0) 
           bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = velocity_y(x[i],y[i],R1,R2,kk,rho0,g0)
           is_outer[i]=True


    NfemTr=np.sum(bc_fix)

    bc_nb=np.zeros(Nfem,dtype=np.int32)  # boundary condition number

    counter=0
    for i in range(0,Nfem):
      if (bc_fix[i]):
         bc_nb[i]=counter
         counter+=1

    print("defining boundary conditions (%.3fs)" % (time.time() - start))

    #################################################################
    # build FE matrix
    #################################################################
    start = time.time()

    a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
    rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    N     = np.zeros(m,dtype=np.float64)            # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
    v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
    k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    vr    = np.zeros(nnp,dtype=np.float64)
    vt    = np.zeros(nnp,dtype=np.float64)

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el = np.zeros(m * ndof)
        a_el = np.zeros((m * ndof, m * ndof), dtype=float)

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
                    b_el[2*i  ]+=N[i]*jcob*wq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
                    b_el[2*i+1]+=N[i]*jcob*wq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)

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
                if bc_fix[m1]: 
                   fixt=bc_val[m1]
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

    np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("reshape solution (%.3fs)" % (time.time() - start))

    for i in range(nnp):
        vr[i]=v[i]*np.sin(theta[i])+u[i]*np.cos(theta[i])
        vt[i]=v[i]*np.cos(theta[i])-u[i]*np.sin(theta[i])

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
        for k in range(0,m):
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

    print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
    print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
    np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute p & sr | time: %.3f s" % (time.time() - start))



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

    #################################################################
    ## Surface C->N
    #################################################################

    txn1=np.zeros(nnp,dtype=np.float64)
    tyn1=np.zeros(nnp,dtype=np.float64)

    for i in range(nnp):
        if bc_fix[2*i]: 
            n_x=x[i]/np.sqrt(x[i]**2+y[i]**2)
            n_y=y[i]/np.sqrt(x[i]**2+y[i]**2)
            if (sqrt(x[i]**2+y[i]**2) > 1.5):

                txn1[i] = n_x*sxxn1[i]+n_y*sxyn1[i]
                tyn1[i] = n_x*sxyn1[i]+n_y*syyn1[i]
            else:
                txn1[i] = -n_x*sxxn1[i]-n_y*sxyn1[i]
                tyn1[i] = -n_x*sxyn1[i]-n_y*syyn1[i]




    #################################################################
    #### CBF
    #################################################################


    M_prime = np.zeros((NfemTr,NfemTr),np.float64)
    rhs_cbf = np.zeros(NfemTr,np.float64)
    tx = np.zeros(nnp,np.float64)
    ty = np.zeros(nnp,np.float64)

    use_ML=True
    if use_ML:
        M_prime_el =np.array([ \
        [1,0],\
        [0,1]])
    else:
        M_prime_el =np.array([ \
        [2./3.,1./3.],\
        [1./3.,2./3.]])

    CBF_use_smoothed_pressure=False

    for iel in range(0,nel):
      #print(iel)
       # set 2 arrays to 0 every loop
      b_el = np.zeros(m * ndof)
      a_el = np.zeros((m * ndof, m * ndof), dtype=float)

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
                  b_el[2*i  ]+=N[i]*jcob*wq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
                  b_el[2*i+1]+=N[i]*jcob*wq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)

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
      for i in range(0,ndof):
          x_difference=x[icon[0,iel]]-x[icon[1,iel]]
          y_difference=y[icon[0,iel]]-y[icon[1,iel]]
          distance=np.sqrt(x_difference**2+y_difference**2)
          idof0=2*icon[0,iel]+i
          idof1=2*icon[1,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[0+i]   
             rhs_cbf[idofTr1]+=rhs_el[2+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*distance/2
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*distance/2
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*distance/2
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*distance/2

      #boundary 1-2 : x,y dofs
      for i in range(0,ndof):
          x_difference=x[icon[2,iel]]-x[icon[1,iel]]
          y_difference=y[icon[2,iel]]-y[icon[1,iel]]
          distance=np.sqrt(x_difference**2+y_difference**2)
          idof0=2*icon[1,iel]+i
          idof1=2*icon[2,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[2+i]   
             rhs_cbf[idofTr1]+=rhs_el[4+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*distance/2
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*distance/2
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*distance/2
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*distance/2

      #boundary 2-3 : x,y dofs
      for i in range(0,ndof):
          x_difference=x[icon[2,iel]]-x[icon[3,iel]]
          y_difference=y[icon[2,iel]]-y[icon[3,iel]]
          distance=np.sqrt(x_difference**2+y_difference**2)
          idof0=2*icon[2,iel]+i
          idof1=2*icon[3,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[4+i]   
             rhs_cbf[idofTr1]+=rhs_el[6+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*distance/2
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*distance/2
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*distance/2
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*distance/2

      #boundary 3-0 : x,y dofs
      for i in range(0,ndof):
          x_difference=x[icon[0,iel]]-x[icon[3,iel]]
          y_difference=y[icon[0,iel]]-y[icon[3,iel]]
          distance=np.sqrt(x_difference**2+y_difference**2)
          idof0=2*icon[3,iel]+i
          idof1=2*icon[0,iel]+i
          if (bc_fix[idof0] and bc_fix[idof1]):  
             idofTr0=bc_nb[idof0]   
             idofTr1=bc_nb[idof1]
             rhs_cbf[idofTr0]+=rhs_el[6+i]   
             rhs_cbf[idofTr1]+=rhs_el[0+i]   
             M_prime[idofTr0,idofTr0]+=M_prime_el[0,0]*distance/2
             M_prime[idofTr0,idofTr1]+=M_prime_el[0,1]*distance/2
             M_prime[idofTr1,idofTr0]+=M_prime_el[1,0]*distance/2
             M_prime[idofTr1,idofTr1]+=M_prime_el[1,1]*distance/2

    print("     -> M_prime (m,M) %.4e %.4e " %(np.min(M_prime),np.max(M_prime)))
    print("     -> rhs_cbf (m,M) %.4e %.4e " %(np.min(rhs_cbf),np.max(rhs_cbf)))

    sol=sps.linalg.spsolve(sps.csr_matrix(M_prime),rhs_cbf)#,use_umfpack=True)

    tx_analytical=np.zeros(nnp,np.float64)
    ty_analytical=np.zeros(nnp,np.float64)
    for i in range(0,nnp):
      idof=2*i+0
      if bc_fix[idof]:
         tx[i]=sol[bc_nb[idof]] 
         tx_analytical[i]=get_analytical_tx(x[i],y[i])
      idof=2*i+1
      if bc_fix[idof]:
         ty[i]=sol[bc_nb[idof]]
         ty_analytical[i]=get_analytical_ty(x[i],y[i])




    np.savetxt("tractions_x.ascii",tx)
    np.savetxt("tractions_y.ascii",ty)

    # fig,ax=plt.subplots()
    # ax.plot(tx,label="cbf")
    # ax.plot(tx_analytical,label="analytical")
    # ax.legend()
    # fig.savefig("traction_x.pdf")

    # fig,ax=plt.subplots()
    # ax.plot(ty,label="cbf")
    # ax.plot(ty_analytical,label="analytical")
    # ax.legend()
    # fig.savefig("traction_y.pdf")

    n_inner=np.sum(is_inner)
    n_outer=np.sum(is_outer)

    tx_inner=np.zeros(n_inner,dtype=np.float64)
    ty_inner=np.zeros(n_inner,dtype=np.float64)
    tx_outer=np.zeros(n_outer,dtype=np.float64)
    ty_outer=np.zeros(n_outer,dtype=np.float64)

    tx_innern1=np.zeros(n_inner,dtype=np.float64)
    ty_innern1=np.zeros(n_inner,dtype=np.float64)
    tx_outern1=np.zeros(n_outer,dtype=np.float64)
    ty_outern1=np.zeros(n_outer,dtype=np.float64)

    tx_inner_analytical=np.zeros(n_inner,dtype=np.float64)
    ty_inner_analytical=np.zeros(n_inner,dtype=np.float64)
    tx_outer_analytical=np.zeros(n_outer,dtype=np.float64)
    ty_outer_analytical=np.zeros(n_outer,dtype=np.float64)

    counter_inner=0
    counter_outer=0
    for i in range(nnp):
        if is_inner[i]:
            tx_inner[counter_inner] = tx[i]
            ty_inner[counter_inner] = ty[i]
            tx_innern1[counter_inner] = txn1[i]
            ty_innern1[counter_inner] = tyn1[i]
            tx_inner_analytical[counter_inner] = get_analytical_tx(x[i],y[i])
            ty_inner_analytical[counter_inner] = get_analytical_ty(x[i],y[i])
            counter_inner+=1
        if is_outer[i]:
            tx_outer[counter_outer] = tx[i]
            ty_outer[counter_outer] = ty[i]
            tx_outern1[counter_outer] = txn1[i]
            ty_outern1[counter_outer] = tyn1[i]
            tx_outer_analytical[counter_outer] = get_analytical_tx(x[i],y[i])
            ty_outer_analytical[counter_outer] = get_analytical_ty(x[i],y[i])
            counter_outer+=1
    fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,figsize=(10,10))

    ax0.plot(tx_inner,label="CBF")
    ax0.plot(tx_innern1,label="n1")
    ax0.plot(tx_inner_analytical,label="analytical")
    ax0.legend()
    ax0.set_title("$t_x$ (inner)")

    ax1.plot(ty_inner,label="CBF")
    ax1.plot(ty_innern1,label="n1")
    ax1.plot(ty_inner_analytical,label="analytical")
    ax1.legend()
    ax1.set_title("$t_y$ (inner)")

    ax2.plot(tx_outer,label="CBF")
    ax2.plot(tx_outern1,label="n1")
    ax2.plot(tx_outer_analytical,label="analytical")
    ax2.legend()
    ax2.set_title("$t_x$ (outer)")

    ax3.plot(ty_outer,label="CBF")
    ax3.plot(ty_outern1,label="n1")
    ax3.plot(ty_outer_analytical,label="analytical")
    ax3.legend()
    ax3.set_title("$t_y$ (outer)")

    fig.savefig("tractions.pdf")

    #######################################################################
    ## Norm Calculations
    #######################################################################

    L1_tx_inner=0
    L1_ty_inner=0
    L1_tx_outer=0
    L1_ty_outer=0

    L2_tx_inner=0
    L2_ty_inner=0
    L2_tx_outer=0
    L2_ty_outer=0

    L1_tx_innern1=0
    L1_ty_innern1=0
    L1_tx_outern1=0
    L1_ty_outern1=0

    L2_tx_innern1=0
    L2_ty_innern1=0
    L2_tx_outern1=0
    L2_ty_outern1=0


    for iel in range(nel):
        el_is_on_inner=False
        el_is_on_outer=False
        for k in range(m):
            if is_inner[icon[k,iel]]:
                el_is_on_inner=True
            if is_outer[icon[k,iel]]:
                el_is_on_outer=True

        if el_is_on_inner:

            distance = np.sqrt((x[icon[0,iel]]-x[icon[1,iel]])**2+(y[icon[0,iel]]-y[icon[1,iel]])**2)

            for iq in [-1,1]:
                rq=iq/sqrt3

                N0=0.5*(1-rq)
                N1=0.5*(1+rq)
                wq=1*1

                xq=N0*x[icon[0,iel]]+N1*x[icon[1,iel]]
                yq=N0*y[icon[0,iel]]+N1*y[icon[1,iel]]

                txq = N0*tx[icon[0,iel]]+N1*tx[icon[1,iel]]
                tyq = N0*ty[icon[0,iel]]+N1*ty[icon[1,iel]]

                L1_tx_inner += abs(txq- get_analytical_tx(xq,yq))*wq*distance
                L1_ty_inner += abs(tyq- get_analytical_ty(xq,yq))*wq*distance

                L2_tx_inner += (txq- get_analytical_tx(xq,yq))**2*wq*distance
                L2_ty_inner += (tyq- get_analytical_ty(xq,yq))**2*wq*distance

                txqn1 = N0*txn1[icon[0,iel]]+N1*txn1[icon[1,iel]]
                tyqn1 = N0*tyn1[icon[0,iel]]+N1*tyn1[icon[1,iel]]

                L1_tx_innern1 += abs(txqn1- get_analytical_tx(xq,yq))*wq*distance
                L1_ty_innern1 += abs(tyqn1- get_analytical_ty(xq,yq))*wq*distance

                L2_tx_innern1 += (txqn1- get_analytical_tx(xq,yq))**2*wq*distance
                L2_ty_innern1 += (tyqn1- get_analytical_ty(xq,yq))**2*wq*distance

        if el_is_on_outer:

            distance = np.sqrt((x[icon[2,iel]]-x[icon[3,iel]])**2+(y[icon[2,iel]]-y[icon[3,iel]])**2)

            for iq in [-1,1]:
                rq=iq/sqrt3

                N0=0.5*(1-rq)
                N1=0.5*(1+rq)
                wq=1*1

                xq=N0*x[icon[2,iel]]+N1*x[icon[3,iel]]
                yq=N0*y[icon[2,iel]]+N1*y[icon[3,iel]]

                txq = N0*tx[icon[2,iel]]+N1*tx[icon[3,iel]]
                tyq = N0*ty[icon[2,iel]]+N1*ty[icon[3,iel]]

                L1_tx_outer += abs(txq- get_analytical_tx(xq,yq))*wq*distance
                L1_ty_outer += abs(tyq- get_analytical_ty(xq,yq))*wq*distance

                L2_tx_outer += (txq- get_analytical_tx(xq,yq))**2*wq*distance
                L2_ty_outer += (tyq- get_analytical_ty(xq,yq))**2*wq*distance

                txqn1 = N0*txn1[icon[2,iel]]+N1*txn1[icon[3,iel]]
                tyqn1 = N0*tyn1[icon[2,iel]]+N1*tyn1[icon[3,iel]]

                L1_tx_innern1 += abs(txqn1- get_analytical_tx(xq,yq))*wq*distance
                L1_ty_innern1 += abs(tyqn1- get_analytical_ty(xq,yq))*wq*distance

                L2_tx_innern1 += (txqn1- get_analytical_tx(xq,yq))**2*wq*distance
                L2_ty_innern1 += (tyqn1- get_analytical_ty(xq,yq))**2*wq*distance


    L1_tx_inner_list.append(L1_tx_inner)
    L1_ty_inner_list.append(L1_ty_inner)
    L2_tx_inner_list.append(np.sqrt(L2_tx_inner))
    L2_ty_inner_list.append(np.sqrt(L2_ty_inner))

    L1_tx_outer_list.append(L1_tx_outer)
    L1_ty_outer_list.append(L1_ty_outer)
    L2_tx_outer_list.append(np.sqrt(L2_tx_outer))
    L2_ty_outer_list.append(np.sqrt(L2_ty_outer))

    L1_tx_inner_listn1.append(L1_tx_innern1)
    L1_ty_inner_listn1.append(L1_ty_innern1)
    L2_tx_inner_listn1.append(np.sqrt(L2_tx_innern1))
    L2_ty_inner_listn1.append(np.sqrt(L2_ty_innern1))

    L1_tx_outer_listn1.append(L1_tx_outern1)
    L1_ty_outer_listn1.append(L1_ty_outern1)
    L2_tx_outer_listn1.append(np.sqrt(L2_tx_outern1))
    L2_ty_outer_listn1.append(np.sqrt(L2_ty_outern1))


hr_list=1/(np.array(nelr_list)+1)
log_hr=np.log(np.abs(hr_list))

fig_x,ax_x=plt.subplots()
fig_y,ax_y=plt.subplots()

# print(len(hr_list))
# print(hr_list)
# print(len(L1_tx_inner_list))
# print(L1_tx_inner_list)
# ax_x.plot(log_hr,np.log(np.array(L1_tx_inner_list)),label="L1 tx inner")
# ax_y.plot(log_hr,np.log(np.array(L1_ty_inner_list)),label="L1 ty inner")
ax_x.plot(log_hr,np.log(np.array(L2_tx_inner_list)),label="L2 tx inner")
ax_y.plot(log_hr,np.log(np.array(L2_ty_inner_list)),label="L2 ty inner")

# ax_x.plot(log_hr,np.log(np.array(L1_tx_outer_list)),label="L1 tx outer")
# ax_y.plot(log_hr,np.log(np.array(L1_ty_outer_list)),label="L1 ty outer")
ax_x.plot(log_hr,np.log(np.array(L2_tx_outer_list)),label="L2 tx outer")
ax_y.plot(log_hr,np.log(np.array(L2_ty_outer_list)),label="L2 ty outer")

# ax_x.plot(log_hr,np.log(np.array(L1_tx_inner_listn1)),label="L1 tx inner C->N")
# ax_y.plot(log_hr,np.log(np.array(L1_ty_inner_listn1)),label="L1 ty inner C->N")
ax_x.plot(log_hr,np.log(np.array(L2_tx_inner_listn1)),label="L2 tx inner C->N")
ax_y.plot(log_hr,np.log(np.array(L2_ty_inner_listn1)),label="L2 ty inner C->N")

# ax_x.plot(log_hr,np.log(np.array(L1_tx_outer_listn1)),label="L1 tx outer C->N")
# ax_y.plot(log_hr,np.log(np.array(L1_ty_outer_listn1)),label="L1 ty outer C->N")
ax_x.plot(log_hr,np.log(np.array(L2_tx_outer_listn1)),label="L2 tx outer C->N")
ax_y.plot(log_hr,np.log(np.array(L2_ty_outer_listn1)),label="L2 ty outer C->N")

ax_x.legend()
ax_y.legend()
fig_x.savefig("convergence_x.pdf")
fig_y.savefig("convergence_y.pdf")


print("L1 tx inner gradient = %10f" % get_regression(hr_list,L1_tx_inner_list))
print("L1 ty inner gradient = %10f" % get_regression(hr_list,L1_ty_inner_list))
print("L2 tx inner gradient = %10f" % get_regression(hr_list,L2_tx_inner_list))
print("L2 ty inner gradient = %10f" % get_regression(hr_list,L2_ty_inner_list))

print("L1 tx outer gradient = %10f" % get_regression(hr_list,L1_tx_outer_list))
print("L1 ty outer gradient = %10f" % get_regression(hr_list,L1_ty_outer_list))
print("L2 tx outer gradient = %10f" % get_regression(hr_list,L2_tx_outer_list))
print("L2 ty outer gradient = %10f" % get_regression(hr_list,L2_ty_outer_list))


#####################################################################
# plot of solution
#####################################################################
start = time.time()

if True:
   vtufile=open("solution.vtu","w")
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
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pressure(xc[iel],yc[iel],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % (p[iel]-pressure(xc[iel],yc[iel],R1,R2,kk,rho0,g0)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %exx[iel])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %eyy[iel])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %exy[iel])
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sxx' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %(2*exx[iel]-p[iel]))
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='syy' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %(2*eyy[iel]-p[iel]))
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sxy' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %(2*exy[iel]))
   vtufile.write("</DataArray>\n")

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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(velocity_x(x[i],y[i],R1,R2,kk,rho0,g0),velocity_y(x[i],y[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(x[i],y[i],R1,R2,kk,rho0,g0),v[i]-velocity_y(x[i],y[i],R1,R2,kk,rho0,g0),0.))
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
       vtufile.write("%10f \n" %density(x[i],y[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='x' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %x[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='tx' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %tx[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='ty' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %ty[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='y' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %y[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='v r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %vr[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='v theta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %vt[i])
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sxx analytical' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %s_xx(x[i],y[i]))
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='syy analytical' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %s_yy(x[i],y[i]))
   vtufile.write("</DataArray>\n")
    #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='sxy analytical' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %s_xy(x[i],y[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='tx analytical' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %get_analytical_tx(x[i],y[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='ty analytical' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %get_analytical_ty(x[i],y[i]))
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
   print("export to vtu | time: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
