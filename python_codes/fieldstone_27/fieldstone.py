import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time




#Weir notes:
#consider replacing range with np.arange - may improve speed - or not as it turns out

#------------------------------------------------------------------------------

def density(x,y,y0,rho_alpha):
    lambdaa=1
    k=2*np.pi/lambdaa
    if abs(y-y0)<1e-6:
       val=rho_alpha*np.cos(k*x)
    else:
       val=0.
    return val

def sigmayy_th(x,y0):
    lambdaa=1.
    k=2*np.pi/lambdaa
    val=np.cos(k*x)/np.sinh(k)**2*\
       (k*(1.-y0)*np.sinh(k)*np.cosh(k*y0)\
       -k*np.sinh(k*(1.-y0))\
       +np.sinh(k)*np.sinh(k*y0) )
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

m=4     # number of nodes making up an element
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
   nelx = 64
   nely = 64
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

gx=0
gy=-1

pnormalise=True
viscosity=1  # dynamic viscosity \mu
y0=63./64.
rho_alpha=64.

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
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]<eps:
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# building density array
#################################################################
rho = np.empty(nnp, dtype=np.float64)  

for i in range(0,nnp):
    rho[i]=density(x[i],y[i],y0,rho_alpha)

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
            rhoq=0.
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                rhoq+=N[k]*rho[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*rhoq*gx
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*rhoq*gy
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

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
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]

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

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# attempt CBF
######################################################################



M_prime = np.zeros((64,64))

rhs_cbf = np.zeros(64)

counter = 0
for iel in range(0,iel):
  if (y[icon[3,iel]] > Ly-eps): #Check if element is on boundary
    on_Upper_Boundary = True
    counter +=1



    # Create the elemental finite element matrixes
    M_prime_el = np.array([[1/3.,1/6.],[1/6.,1/3.]])

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
            rhoq=0.
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                rhoq+=N[k]*rho[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*rhoq*gx
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*rhoq*gy
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

    vx_el = [u[icon[0,iel]],u[icon[1,iel]],u[icon[2,iel]],u[icon[3,iel]]]
    vy_el = [v[icon[0,iel]],v[icon[1,iel]],v[icon[2,iel]],v[icon[3,iel]]]

    v_el = np.array([vx_el[0],vy_el[0],vx_el[1],vy_el[1],vx_el[2],vy_el[2],vx_el[3],vy_el[3]])

    #K_prime_el
    K_prime_el = np.array([np.dot(K_el[:,5],v_el),np.dot(K_el[:,7],v_el)])

    #print(K_prime_el.shape)


    #g_prime_el

    p_el = p[iel]
    g_prime_el = np.array([G_el[5]*p_el,G_el[7]*p_el])
    #print(g_prime_el.shape)


    #f_prime_el
    f_prime_el = np.array([f_el[5],f_el[7]])
    #print(f_prime_el.shape)

    #Now assemble the matrixes

    M_prime[counter-1:counter+1,counter-1:counter+1] += M_prime_el*hx
    #print(rhs_cbf[counter-1:counter+1].shape)
    #print((-K_prime_el - g_prime_el + f_prime_el).shape)
    rhs_cbf[counter-1] += K_prime_el[0] + g_prime_el[0] - f_prime_el[0]
    rhs_cbf[counter]   += K_prime_el[1] + g_prime_el[1] - f_prime_el[1]


tractions=sps.linalg.spsolve(sps.csr_matrix(M_prime),rhs_cbf)

np.savetxt("tractions.dat",tractions)

print(tractions[0],sigmayy_th(0,y0),(tractions[0]-sigmayy_th(0,y0))/sigmayy_th(0,y0)*100 )


######################################################################
# compute nodal pressure
######################################################################

q=np.zeros(nnp,dtype=np.float64)  
eyyn=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    q[icon[3,iel]]+=p[iel]
    eyyn[icon[0,iel]]+=eyy[iel]
    eyyn[icon[1,iel]]+=eyy[iel]
    eyyn[icon[2,iel]]+=eyy[iel]
    eyyn[icon[3,iel]]+=eyy[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1

q/=count
eyyn/=count

np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

######################################################################
# export elemental sigma_yy
######################################################################
sigmayy_el = np.empty(nel, dtype=np.float64)  
sigmayy_analytical = np.empty(nnx,dtype=np.float64)  

for iel in range(1,nel):
    sigmayy_el[iel]=(-p[iel]+2.*viscosity*eyy[iel])/abs(gy)

np.savetxt('sigmayy_el.ascii',np.array([xc[nel-nelx:nel],\
                                        sigmayy_el[nel-nelx:nel],\
                                        p[nel-nelx:nel],\
                                        2.*viscosity*eyy[nel-nelx:nel],\
                                       ]).T,header='# xc,sigmayy')

np.savetxt('sigmayy_nod.ascii',np.array([x[nnp-nnx:nnp],\
                                        -q[nnp-nnx:nnp]+2.*viscosity*eyyn[nnp-nnx:nnp],\
                                        -q[nnp-nnx:nnp]+2.*viscosity*eyyn[nnp-nnx:nnp] - \
                                         sigmayy_th(x[nnp-nnx:nnp],y0),\
                                        ]).T,header='# x,sigmayy,error')

counter=0
for i in range(nnp-nnx,nnp):
    sigmayy_analytical[counter]=sigmayy_th(x[i],y0)
    counter+=1

np.savetxt('sigmayy_analytical.ascii',np.array([x[nnp-nnx:nnp],sigmayy_analytical]).T,header='# x,sigmayy')

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
vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % sigmayy_el[iel])
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
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %rho[i])
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")


