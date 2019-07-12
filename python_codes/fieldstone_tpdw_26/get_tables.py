import numpy as np
from scipy import stats
def get_regression(h,y):
    y=np.abs(y)
    x=np.abs(h)
    #return np.linalg.lstsq(np.vstack([np.log(h), np.ones(len(np.log(h)))]).T,np.log(y))[0][0]
    return stats.linregress(np.log(x),np.log(y))[0]

exx_l2 =np.loadtxt("errors_exx",usecols=[2,5,6,7,8,9,10,11,12])
exy_l2 =np.loadtxt("errors_exy",usecols=[2,5,6,7,8,9,10,11,12])
exx_int=np.loadtxt("nodal_errors_exx_internal",usecols=[2,8,9,10])
exy_int=np.loadtxt("nodal_errors_exy_internal",usecols=[2,8,9,10])
exx_edg=np.loadtxt("nodal_errors_exx_edge",usecols=[2,8,9,10])
exy_edg=np.loadtxt("nodal_errors_exx_edge",usecols=[2,8,9,10])

exx_l2_centre =get_regression(1/np.sqrt(exx_l2[:,0] ),exx_l2[:,2] )
exx_l2_corner =get_regression(1/np.sqrt(exx_l2[:,0] ),exx_l2[:,3] )
exx_l2_superc =get_regression(1/np.sqrt(exx_l2[:,0] ),exx_l2[:,8] )
exy_l2_centre =get_regression(1/np.sqrt(exy_l2[:,0] ),exy_l2[:,2] )
exy_l2_corner =get_regression(1/np.sqrt(exy_l2[:,0] ),exy_l2[:,3] )
exy_l2_superc =get_regression(1/np.sqrt(exy_l2[:,0] ),exy_l2[:,8] )

exx_int_centre=get_regression(1/np.sqrt(exx_int[:,0]),exx_int[:,1])
exy_int_centre=get_regression(1/np.sqrt(exy_int[:,0]),exy_int[:,1])
exx_int_corner=get_regression(1/np.sqrt(exx_int[:,0]),exx_int[:,2])
exy_int_corner=get_regression(1/np.sqrt(exy_int[:,0]),exy_int[:,2])
exx_int_superc=get_regression(1/np.sqrt(exx_int[:,0]),exx_int[:,3])
exy_int_superc=get_regression(1/np.sqrt(exy_int[:,0]),exy_int[:,3])

exx_edg_centre=get_regression(1/np.sqrt(exx_edg[:,0]),exx_edg[:,1])
exy_edg_centre=get_regression(1/np.sqrt(exy_edg[:,0]),exy_edg[:,1])
exx_edg_corner=get_regression(1/np.sqrt(exx_edg[:,0]),exx_edg[:,2])
exy_edg_corner=get_regression(1/np.sqrt(exy_edg[:,0]),exy_edg[:,2])
exx_edg_superc=get_regression(1/np.sqrt(exx_edg[:,0]),exx_edg[:,3])
exy_edg_superc=get_regression(1/np.sqrt(exy_edg[:,0]),exy_edg[:,3])

filename="table_q2q1_convergences"
file=open(filename,'w')
file.write("\\begin{table}\n")
file.write("\label{table:q2q1}\n")
file.write("\caption{Top left corner tractions.}\n")
file.write("\\begin{center}\n")
file.write("\\begin{tabular}{| c | c c c c c c |} \n")
file.write("\hline \n")
file.write("Method & \multicolumn{6}{c|}{Convergence Rate} \\\ \hline\n")
file.write("& \multicolumn{2}{c|}{$L_2$} & \multicolumn{2}{c|}{Interal} & \multicolumn{2}{c|}{Edge} \\\ \hline\n")
file.write("& $e_{xx}$ & $e_{xy}$ & $e_{xx}$ & $e_{xy}$ & $e_{xx}$ & $e_{xy}$ \\\ \hline\n")
file.write("Centre to Node & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\ \hline\n".format(exx_l2_centre,exy_l2_centre,exx_int_centre,exy_int_centre,exx_edg_centre,exy_edg_centre))
file.write("Corner to Node & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\ \hline\n".format(exx_l2_corner,exy_l2_corner,exx_int_corner,exy_int_corner,exx_edg_corner,exy_edg_corner))
file.write("Superconvergent Patch Recovery & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\ \hline\n".format(exx_l2_superc,exy_l2_superc,exx_int_superc,exy_int_superc,exx_edg_superc,exy_edg_superc))
file.write("\end{tabular}\n")
file.write("\end{center}\n")
file.write("\end{table}\n")
file.close()