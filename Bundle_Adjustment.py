import numpy as np
from Reprojection_Error import OptimReprojectionError
from scipy.optimize import least_squares

def BundleAdjustment(points_3d, temp2, Rt2, K, r_error):
	
	opt_variables_init = np.hstack((Rt2.ravel(), K.ravel()))
	opt_variables_2d = np.hstack((opt_variables_init, temp2.ravel()))			
	n2d = int(opt_variables_2d.shape[0] - opt_variables_init.shape[0])
	opt_variables = np.hstack((opt_variables_2d, points_3d.ravel()))
	corrected_values = least_squares(fun = OptimReprojectionError, x0 = opt_variables, args=(n2d,),  gtol = r_error).x
	Rt = corrected_values[0:12].reshape((3,4))
	K = corrected_values[12:21].reshape((3,3))
	p = corrected_values[21:21 + n2d].reshape((2, int(n2d/2))).T
	X = corrected_values[21 + n2d:].reshape((int(len(corrected_values[21 + n2d:])/3), 3))

	return X, p, Rt