import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import math
from matplotlib.widgets import LassoSelector
from matplotlib import path


from scipy.interpolate import griddata


class LassoSelect:
    def __init__(self, xyz):

        # width = np.size(grid_z, axis=1)
        # height = np.size(grid_z, axis=0)
        # self.shape = (height, width)
        # self.array = np.zeros(self.shape)
        #
        # x, y = np.meshgrid(np.arange(width), np.arange(height))
        #self.pix = np.vstack((x.flatten(), y.flatten())).T

        grid_z, _ = interpolate(xyz)

        #Von Mittelwert befreien und auf Bild-Indices anpassen, da sich Lasso auf Bild-Koordinaten bezieht
        xyz[:, 0:2] = xyz[:, 0:2] - [np.min(xyz[:, 0]), np.min(xyz[:, 1])]
        scale = np.size(grid_z, axis=0) / (np.max(xyz[:, 0]) - np.min(xyz[:, 0]))
        self.points = xyz[:, 0:2] * scale

        #print(np.max(self.points))


        self.fig, ax = plt.subplots(1, 1)
        ax.imshow(grid_z.T, cmap='turbo')

        lasso = LassoSelector(ax, self.onselect)
        plt.show()

    def onselect(self, verts):
        #global array, pix
        p = path.Path(verts)
        self.ind = p.contains_points(self.points, radius=1)
        #lin = np.arange(self.array.size)
        #newArray = self.array.flatten()
        #newArray[lin[self.ind]] = 1
        #self.selection = newArray#.reshape(self.shape) #, order='F'
        self.fig.canvas.draw_idle()

def sphereFit(xyz, n_sigma, iterations):
    #   Assemble the A matrix
    spX = xyz[:, 0]
    spY = xyz[:, 1]
    spZ = xyz[:, 2]

    for k in range(iterations + 1):

        A = np.zeros((len(spX),4))
        A[:,0] = spX*2
        A[:,1] = spY*2
        A[:,2] = spZ*2
        A[:,3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX),1))
        f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
        C, residules, rank, singval = np.linalg.lstsq(A,f)

        #   solve for the radius
        t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
        r = math.sqrt(t.item())
        xc = C[0]
        yc = C[1]
        zc = C[2]

        c = np.asarray([xc, yc, zc]).T
        spXYZ = np.asarray([spX, spY, spZ]).T

        dists = np.linalg.norm(spXYZ - c, axis=1) - r
        std = np.std(dists)

        print(min(dists))
        print(max(dists))

        s_inds = np.where(abs(dists) < n_sigma * std)

        spX = spX[s_inds]
        spY = spY[s_inds]
        spZ = spZ[s_inds]
        dists = dists[s_inds]

        spXYZ = np.asarray([spX, spY, spZ]).T

    return r, C[0], C[1], C[2], dists, spXYZ

def planeFit(xyz, n_sigma, iterations):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    for k in range(iterations + 1):
    # least squares
        A = np.array((np.ones_like(x), x, y)).T
        b = z.T
        plane_params = np.linalg.inv(A.T @ A) @ A.T @ b

        a = plane_params[1]
        b = plane_params[2]
        c = -1
        d = plane_params[0]

        dists = ((a * x + b * y + c * z + d) /
             math.sqrt(a ** 2 + b ** 2 + c ** 2))

        print(min(dists))
        print(max(dists))

        std = np.std(dists)

        s_inds = np.where(abs(dists) < n_sigma * std)

        x = x[s_inds]
        y = y[s_inds]
        z = z[s_inds]
        dists = dists[s_inds]

    return a, b, c, d, dists

def damage_calculation(xyz, mask):

    inds_plane = np.where(mask == False)[0]
    inds_sphere = np.where(mask == True)[0]

    plane_points = xyz[inds_plane, :]
    sphere_points = xyz[inds_sphere, :]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2])
    # plt.show()

    a, b, c, d, _ = planeFit(plane_points, 3, 1)

    dists = ((a * sphere_points[:, 0] + b * sphere_points[:, 1] + c * sphere_points[:, 2] + d) /
             math.sqrt(a ** 2 + b ** 2 + c ** 2))

    inds_sphere = np.where(abs(dists) > 0.05)[0]

    sphere_points = sphere_points[inds_sphere, :]

    print('d' + str(np.max(sphere_points[:, 0])- np.min(sphere_points[:, 0])))

    # interpolate(plane_points, vis=True)
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2])
    # plt.show()

    radius_sphere, xc, yc, zc, sphere_dists, sphere_points_inliners = sphereFit(sphere_points, 3, 1)

    interpolate(sphere_points_inliners, vis=True, dists=sphere_dists)

    dist_c = ((a * xc + b * yc + c * zc + d) /
             math.sqrt(a ** 2 + b ** 2 + c ** 2))

    depth = radius_sphere - abs(dist_c.item())
    radius =  math.sqrt(radius_sphere ** 2 - dist_c.item() ** 2)

    return depth, radius, radius_sphere

def interpolate(xyz, vis=True, dists=None):

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xy = np.vstack((x, y)).T

    res = 0.01
    grid_x, grid_y = np.mgrid[min(x):max(x):res, min(y):max(y):res]
    print('interpolating....')
    grid_z = griddata(xy, z, (grid_x, grid_y), method='nearest')


    if vis:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        if dists is not None:
            grid_dists = griddata(xy, dists, (grid_x, grid_y), method='nearest')
            #grid_dists[grid_dists == np.nan] = 0

            scmap = plt.cm.ScalarMappable(cmap='turbo')
            scmap.set_array(grid_dists)
            scmap.set_clim(vmin=1 * np.min(grid_dists), vmax=1 * np.max(grid_dists))
            ax.plot_surface(grid_x, grid_y, grid_z, facecolors=scmap.to_rgba(grid_dists), rstride=1,cstride=1, linewidth=0, antialiased=False)
            # , rstride=1, cstride=1, facecolors=my_col, linewidth=0, antialiased=False)#cmap=mappable.cmap, norm=mappable.norm,
            plt.colorbar(scmap, ax=ax)

        else:
            ax.plot_surface(grid_x, grid_y, grid_z)

        plt.gca().set_aspect('equal')
        #ax.set_zlim(-1.01, 1.01)
        plt.show()

    grid_xf = grid_x.flatten()
    grid_yf = grid_y.flatten()
    grid_zf = grid_z.flatten()

    xyz = np.vstack((grid_xf, grid_yf, grid_zf)).T

    return grid_z, xyz


if __name__ == '__main__':

    n = 1
    depth = np.zeros(n)
    radius = np.zeros(n)

    for i in range(n):
        #xyz = genfromtxt("D:\RoboKop\Messdaten\loch2.txt", delimiter=',')
        xyz = np.load(r"W:\Robokop\Messdaten\XYZ\loch1_" +str(i) + ".npy")
        xm = 0.2
        ym = 0.5
        a = 5
        xmax = xm + 0.5 * a
        xmin = xm - 0.5 * a
        ymax = ym + 0.5 * a
        ymin = ym - 0.5 * a

        roi = (xyz[:, 0] < xmax) * (xyz[:, 0] > xmin) * (xyz[:, 1] < ymax) * (xyz[:, 1] > ymin)

        xyz = xyz[roi]
        xyz[:, 2] = -1 * xyz[:, 2]

        lasso = LassoSelect(xyz)
        mask = lasso.ind

        data = damage_calculation(xyz, mask)
        depth[i] = data[0]
        radius[i] = data[1]

    print(np.mean(depth))
    print(np.std(depth))
    print(np.mean(radius))
    print(np.std(radius))

