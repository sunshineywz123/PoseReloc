import torch
from .rotation import *

def DistortV2(xp, yp, cam, cam_K):
	l1 = cam_K[0, 1]
	l2 = cam_K[0, 2]
	focal = cam_K[0, 0]

	# l1 = cam[:, 7]
	# l2 = cam[:, 8]
	# focal = cam[:, 6]

	# FIXME: modify implementation here,
	#  	take cam[:, 7] and cam[:, 8] as cx and cy instead of k1 and k2

	# r2 = xp * xp + yp * yp
	# distortion = 1.0 + r2 * (l1 + l2 * r2)
	# predicted_x = -focal * xp * distortion
	# predicted_y = -focal * yp * distortion

	predicted_x = focal * xp + l1
	predicted_y = focal * yp + l2

	return predicted_x, predicted_y

def Distort(xp, yp, cam):
	l1 = cam[:, 7]
	l2 = cam[:, 8]
	focal = cam[:, 6]

	# FIXME: modify implementation here,
	#  	take cam[:, 7] and cam[:, 8] as cx and cy instead of k1 and k2

	# r2 = xp * xp + yp * yp
	# distortion = 1.0 + r2 * (l1 + l2 * r2)
	# predicted_x = -focal * xp * distortion
	# predicted_y = -focal * yp * distortion

	predicted_x = focal * xp + l1
	predicted_y = focal * yp + l2

	return predicted_x, predicted_y

def SnavelyReprojectionError(points_ob, cameras_ob, features):
	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]
    # camera[0,1,2] are the angle-axis rotation.
	p = AngleAxisRotatePoint(cameras_ob[:, :3], points_ob)
	p = p + cameras_ob[:, 3:6]

	xp = p[:,0] / p[:,2]
	yp = p[:,1] / p[:,2]

	predicted_x, predicted_y = Distort(xp, yp, cameras_ob)

	residual_0 = predicted_x - features[:, 0]
	residual_1 = predicted_y - features[:, 1]

	residual_0 = residual_0.reshape((residual_0.shape[0], 1))
	residual_1 = residual_1.reshape((residual_1.shape[0], 1))

	#return torch.sqrt(residual_0**2 + residual_1 ** 2)
	return torch.cat([residual_0, residual_1], dim=1)

def SnavelyReprojectionErrorV2(points_ob, cameras_ob, features):
	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]
	focals = features[:, 2]
	l1 = features[:, 3]
	l2 = features[:, 4]

    # camera[0,1,2] are the angle-axis rotation.
	p = AngleAxisRotatePoint(cameras_ob[:, :3], points_ob)
	p = p + cameras_ob[:, 3:6]

	xp = p[:,0] / p[:,2]
	yp = p[:,1] / p[:,2]

	# predicted_x, predicted_y = DistortV2(xp, yp, cameras_ob, cam_K)

	predicted_x = focals * xp + l1
	predicted_y = focals * yp + l2

	residual_0 = predicted_x - features[:, 0]
	residual_1 = predicted_y - features[:, 1]

	residual_0 = residual_0.reshape((residual_0.shape[0], 1))
	residual_1 = residual_1.reshape((residual_1.shape[0], 1))

	#return torch.sqrt(residual_0**2 + residual_1 ** 2)
	return torch.cat([residual_0, residual_1], dim=1)

def SnavelyBlockError(points_ob, cameras_ob,
	features, pointMask, pointConstant):

	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]

	points_ob = points_ob * pointMask + pointConstant

    # camera[0,1,2] are the angle-axis rotation.
	p = AngleAxisRotatePoint(cameras_ob[:, :3], points_ob)
	p = p + cameras_ob[:, 3:6]

	xp = p[:,0] / p[:,2]
	yp = p[:,1] / p[:,2]

	predicted_x, predicted_y = Distort(xp, yp, cameras_ob)

	residual_0 = predicted_x - features[:, 0]
	residual_1 = predicted_y - features[:, 1]

	residual_0 = residual_0.reshape((residual_0.shape[0], 1))
	residual_1 = residual_1.reshape((residual_1.shape[0], 1))

	return torch.cat([residual_0, residual_1], dim=1)


def SphericalReprojectionError2D(points_ob, cameras_ob, features):
	# camera[0,1,2] are the angle-axis rotation.

	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]

	P_ob = AngleAxisRotatePoint(cameras_ob[:,:3], points_ob - cameras_ob[:,3:])
	P_ob = Normalize(P_ob)
	P_ob_proj = EquirectangularProjection(P_ob)

	return P_ob_proj - features

def SphericalBlockError2D(points_ob, cameras_ob, features,
	pointMask, pointConstant):
	# camera[0,1,2] are the angle-axis rotation.

	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]

	points_ob = points_ob * pointMask + pointConstant
	P_ob = AngleAxisRotatePoint(cameras_ob[:,:3], points_ob - cameras_ob[:,3:])
	P_ob = Normalize(P_ob)
	P_ob_proj = EquirectangularProjection(P_ob)

	return P_ob_proj - features

def ColmapReprojectionError(points_ob, cameras_ob, features):
	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]
    # camera[0,1,2] are the angle-axis rotation.
	p = AngleAxisRotatePoint(cameras_ob[:, :3], points_ob)
	p = p + cameras_ob[:, 3:6]

	xp = p[:,0] / p[:,2]
	yp = p[:,1] / p[:,2]
	r2 = xp * xp + yp * yp
	r4 = r2 * r2
	r6 = r4 * r2
	coeff = 1 + cameras_ob[:,9] * r2 + cameras_ob[:,10] * r4 + cameras_ob[:,11] * r6

	predicted_x = xp * coeff * cameras_ob[:,6] + cameras_ob[:,7]
	predicted_y = yp * coeff * cameras_ob[:,6] + cameras_ob[:,8]

	residual_0 = predicted_x - features[:, 0]
	residual_1 = predicted_y - features[:, 1]

	residual_0 = residual_0.reshape((residual_0.shape[0], 1))
	residual_1 = residual_1.reshape((residual_1.shape[0], 1))

	#return torch.sqrt(residual_0**2 + residual_1 ** 2)
	return torch.cat([residual_0, residual_1], dim=1)

def RiemannReprojectionError(points_ob, cameras_ob, features):
	# camera[0,1,2] are the angle-axis rotation.

	if (len(points_ob.shape) == 3):
		points_ob = points_ob[:,0,:]
		cameras_ob = cameras_ob[:,0,:]


	P_ob = AngleAxisRotatePoint(cameras_ob[:,:3], points_ob - cameras_ob[:,3:])
	diff_x = (P_ob[:,0] / P_ob[:,2] - features[:,0]).view(-1,1)
	diff_y = (P_ob[:,1] / P_ob[:,2] - features[:,1]).view(-1,1)

	return torch.cat((diff_x, diff_y), dim=1) * 7464.101560
