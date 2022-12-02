import pykitti

# Change this to the directory where you store KITTI data
basedir = 'KITTI_SAMPLE/RAW'

# Specify the dataset to load
date = '2011_09_26'
drive = '0009'

# Load the data. Optionally, specify the frame range to load.
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 447, 1))

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx


#2 Reading and displaying the trajectory

#OXTS inertial unit data
import matplotlib.pyplot as plt

for i in range(0,447,1):
    pose = dataset.oxts[i].T_w_imu[0:3,3] #3d translation vector
    plt.scatter(pose[0], pose[2], s=2, c='b')
plt.title('X & Z IMU')
plt.show()
#
for i in range(0,447,1):
    pose = dataset.oxts[i].T_w_imu[0:3,3] #3d translation vector
    plt.scatter(pose[1], pose[0], s=2, c='b')
plt.title('Y & X IMU')
plt.show()

for i in range(0,447,1):
    pose = dataset.oxts[i].T_w_imu[0:3,3] #3d translation vector
    plt.scatter(pose[1], pose[2], s=2, c='b')
plt.title('Y & Z IMU')
plt.show()

import numpy as np

pxt = np.array([])
pyt = np.array([])
pzt = np.array([])
tz = []
R = np.array([])
G = np.array([])
B = np.array([])

for i in range(50):
    img_cam2 = dataset.get_cam2(i) #load multiple images ( first 50 images)
    cam0 = np.array(img_cam2)
    velo = dataset.get_velo(i) # loading several lidar values
    velo = velo[velo[:, 0] > 5]
    velo[:,3] = 1 #homogeneous 3D coordinates to obtain our 3 points X,Y,Z
    test4 = np.linalg.inv(dataset.calib.T_velo_imu) @ velo.T #transformation mark velo to  imu
    testz = dataset.oxts[i].T_w_imu @ test4 #We get our 3D points with a translation with oxts
    Tp = dataset.calib.T_cam2_velo
    K = dataset.calib.K_cam2
    mat = np.eye(4)
    mat[0:3, 0:3] = K
    T = mat @ Tp
    prod = T @ velo.transpose()  #transition from a 3d  to 2d
    proj = np.array([prod[0] / prod[2], prod[1] / prod[2]]) #the 3D to 2d is done by dividing lines 1 and 2 by the 3rd


# use an np.logical to filter
    mask = ((0 > proj[0]) | (proj[0] > cam0.shape[1] - 1)) | ((0 > proj[1]) | (proj[1] > cam0.shape[0] - 1))
    px = proj[0][mask.ravel() == False]
    py = proj[1][mask.ravel() == False]

    #we convert px and py to int to be able to extract the colors from the cam0 image
    x = py.astype(int)
    y = px.astype(int)


    cam3D0 = cam0[x, y, 0] #extraction of the red color
    R = np.append(R, cam3D0) 
    cam3D1 = cam0[x, y, 1] #extraction of the green color
    G = np.append(G, cam3D1) 
    cam3D2 = cam0[x, y, 2] #extraction of the blue color
    B = np.append(B, cam3D2) 

#We will filter the 3D points obtained (test) by applying a mask to them
# then we will add them after with append
    pxt = np.append(pxt, testz[0][mask.ravel() == False], axis=0)
    pyt = np.append(pyt, testz[1][mask.ravel() == False], axis=0)
    pzt = np.append(pzt, testz[2][mask.ravel() == False], axis=0)

#We will put each component in a row with ravel to be able to use a tuple of size 6 (x,y,z,r,g,b)

X = np.ravel(pxt)
Y = np.ravel(pyt)
Z = np.ravel(pzt)
Red = np.ravel(R)
Green = np.ravel(G)
Blue = np.ravel(B)

#creation of a tuple of size 6 that we will use for the 3D reconstruction

import plyfile as ply
pct = list(zip(X, Y, Z, Red, Green, Blue))
vertex = np.array(pct,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
el = ply.PlyElement.describe(vertex, 'vertex')
ply.PlyData([el]).write('binaryt.ply')

