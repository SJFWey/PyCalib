import math

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata


def sphereFit(xyz, n_sigma, iterations):
    spX = xyz[:, 0]
    spY = xyz[:, 1]
    spZ = xyz[:, 2]

    std = np.nan

    for k in range(iterations + 1):
        if spX.size < 4:
            raise ValueError("Not enough points for sphere fit.")

        A = np.zeros((len(spX), 4))
        A[:, 0] = spX * 2
        A[:, 1] = spY * 2
        A[:, 2] = spZ * 2
        A[:, 3] = 1

        f = np.zeros((len(spX), 1))
        f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
        C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

        t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
        if t.item() < 0:
            print(
                "Warning: Negative value encountered in sphere radius calculation. Setting radius to 0."
            )
            r = 0.0
        else:
            r = math.sqrt(t.item())
        xc = C[0]
        yc = C[1]
        zc = C[2]

        c = np.asarray([xc, yc, zc]).T
        spXYZ_current = np.asarray([spX, spY, spZ]).T

        c_flat = c.flatten()
        if spXYZ_current.shape[1] != c_flat.shape[0]:
            raise ValueError(
                f"Dimension mismatch between points ({spXYZ_current.shape}) and center ({c_flat.shape})."
            )

        dists = np.linalg.norm(spXYZ_current - c_flat, axis=1) - r
        std = np.std(dists)

        if std < 1e-9:
            raise ValueError(
                "Standard deviation of sphere distances is near zero. Skipping outlier removal."
            )

        s_inds = np.where(abs(dists) < n_sigma * std)

        if len(s_inds[0]) < 4:
            raise ValueError(
                "Outlier removal step left too few points for sphere fit. Using results from before filtering."
            )

        spX = spX[s_inds]
        spY = spY[s_inds]
        spZ = spZ[s_inds]
        dists = dists[s_inds]

        spXYZ = np.asarray([spX, spY, spZ]).T

    if "spXYZ" not in locals() or spXYZ.shape[0] < 4:
        raise ValueError(
            "Sphere fit did not converge or resulted in too few points. Returning NaN/empty."
        )

    final_dists = np.linalg.norm(spXYZ - c.flatten(), axis=1) - r

    return r, C[0].item(), C[1].item(), C[2].item(), final_dists, spXYZ, std


def planeFit(xyz, n_sigma, iterations):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    std = np.nan

    for k in range(iterations + 1):
        if x.size < 3:
            raise ValueError("Not enough points for plane fit.")

        A = np.array((np.ones_like(x), x, y)).T
        b = z.T
        plane_params = np.linalg.pinv(A.T @ A) @ A.T @ b

        a = plane_params[1]
        b = plane_params[2]
        c = -1
        d = plane_params[0]

        denom = math.sqrt(a**2 + b**2 + c**2)
        if denom < 1e-9:
            raise ValueError(
                "Plane normal vector is near zero. Skipping distance calculation."
            )

        dists = (a * x + b * y + c * z + d) / denom

        std = np.std(dists)

        if std < 1e-9:
            raise ValueError(
                "Standard deviation of plane distances is near zero. Skipping outlier removal."
            )

        s_inds = np.where(abs(dists) < n_sigma * std)

        if len(s_inds[0]) < 3:
            raise ValueError(
                "Outlier removal step left too few points for plane fit. Using results from before filtering."
            )

        x = x[s_inds]
        y = y[s_inds]
        z = z[s_inds]
        dists = dists[s_inds]

    if "a" not in locals():
        raise ValueError("Plane fit failed. Returning NaN.")

    final_dists = (a * x + b * y + c * z + d) / math.sqrt(a**2 + b**2 + c**2)

    return a, b, c, d, final_dists, std


def damage_calculation(xyz, mask):
    inds_plane = np.where(~mask)[0]
    inds_sphere = np.where(mask)[0]

    if len(inds_plane) < 3:
        raise ValueError("Not enough points for plane fitting.")
    if len(inds_sphere) < 4:
        raise ValueError("Not enough initial points for sphere fitting.")

    plane_points = xyz[inds_plane, :]
    sphere_points = xyz[inds_sphere, :]

    a, b, c, d, plane_inlier_dists, plane_fit_std = planeFit(plane_points, 3, 1)
    if np.isnan(a):
        raise ValueError("Plane fitting failed.")
    print(f"Plane Fit Residual Std Dev (Inliers): {plane_fit_std:.4f}")

    dists_to_plane = (
        a * sphere_points[:, 0] + b * sphere_points[:, 1] + c * sphere_points[:, 2] + d
    ) / math.sqrt(a**2 + b**2 + c**2)

    plane_dist_threshold = 0.05
    inds_sphere_off_plane = np.where(np.abs(dists_to_plane) > plane_dist_threshold)[0]
    sphere_points_filtered = sphere_points[inds_sphere_off_plane, :]

    if sphere_points_filtered.shape[0] < 4:
        raise ValueError(
            f"Not enough points ({sphere_points_filtered.shape[0]}) remaining for sphere fitting after plane distance filtering (threshold: {plane_dist_threshold})."
        )

    (
        radius_sphere,
        xc,
        yc,
        zc,
        sphere_inlier_dists,
        sphere_points_inliers,
        sphere_fit_std,
    ) = sphereFit(sphere_points_filtered, 3, 1)

    if np.isnan(radius_sphere):
        raise ValueError("Sphere fitting failed.")
    print(f"Sphere Fit Residual Std Dev (Inliers): {sphere_fit_std:.4f}")

    vis_plane_points = plane_points
    vis_sphere_points = sphere_points_inliers

    combined_vis_xyz = np.vstack((vis_plane_points, vis_sphere_points))
    combined_vis_dists = np.concatenate(
        (np.abs(plane_inlier_dists), np.abs(sphere_inlier_dists))
    )

    if (
        combined_vis_dists is not None
        and combined_vis_xyz.shape[0] == combined_vis_dists.shape[0]
    ):
        interpolate(combined_vis_xyz, vis=True, dists=combined_vis_dists)
    else:
        interpolate(xyz, vis=True, dists=None)

    plane_model_norm = math.sqrt(a**2 + b**2 + c**2)
    all_plane_dists = (
        a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d
    ) / plane_model_norm

    sphere_center = np.array([xc, yc, zc])
    all_sphere_dists = np.linalg.norm(xyz - sphere_center, axis=1) - radius_sphere

    global_dists = np.where(mask, all_sphere_dists, all_plane_dists)
    abs_global_dists = np.abs(global_dists)

    global_rmse = np.sqrt(np.mean(np.square(global_dists)))
    global_mae = np.mean(abs_global_dists)
    global_max_dev = np.max(abs_global_dists)

    print(f"Global RMSE: {global_rmse:.4f}")
    print(f"Global MAE: {global_mae:.4f}")
    print(f"Global Max Deviation: {global_max_dev:.4f}")

    dist_c = (a * xc + b * yc + c * zc + d) / plane_model_norm

    depth = radius_sphere - abs(dist_c)
    radius_intersection = 0
    if radius_sphere**2 >= dist_c**2:
        radius_intersection = math.sqrt(radius_sphere**2 - dist_c**2)
    else:
        raise ValueError(
            f"Sphere center distance to plane ({abs(dist_c):.4f}) > radius ({radius_sphere:.4f}). No intersection."
        )

    return depth, radius_intersection, radius_sphere


def interpolate(xyz, interp_method="nearest", vis=False, dists=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]  # (N, 3)

    xy = np.vstack((x, y)).T

    if vis:
        if xyz.shape[0] == 0:
            raise ValueError("Cannot visualize empty point cloud.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        if dists is not None and dists.shape[0] == xyz.shape[0]:
            vmin = 0
            vmax = np.max(dists) * 0.9
            if vmin >= vmax:
                vmax = vmin + 1e-6

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("viridis")

            point_colors = cmap(norm(dists))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(point_colors)

            fig, ax = plt.subplots(figsize=(1.5, 6))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax)
            cbar.set_label("Distance (deviation)")
            fig.suptitle("Color Scale", fontsize=10, y=0.95)
            fig.tight_layout(rect=[0, 0, 0.8, 0.9])
            plt.show(block=False)
        else:
            if dists is not None:
                print(
                    "Warning: Mismatch between points and distance data size. Using default color."
                )
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        o3d.visualization.draw_geometries(
            [pcd], window_name="Point Cloud", point_show_normal=False
        )

    res = 0.01
    if x.size > 0 and y.size > 0:
        grid_x, grid_y = np.mgrid[min(x) : max(x) : res, min(y) : max(y) : res]
        if grid_x.size > 0 and grid_y.size > 0:
            grid_z = griddata(xy, z, (grid_x, grid_y), method=interp_method)
            grid_z = np.nan_to_num(grid_z)
        else:
            print("Warning: Grid creation resulted in empty grid in interpolate.")
            grid_z = np.array([[]])
    else:
        print("Warning: Empty input coordinates for interpolation.")
        grid_z = np.array([[]])

    return grid_z, xyz


def process_roi(xyz_original, roi_params, ransac_params):
    xm = roi_params.get("xm", 0)
    ym = roi_params.get("ym", 0)
    a = roi_params.get("a", 1)
    xmax = xm + 0.5 * a
    xmin = xm - 0.5 * a
    ymax = ym + 0.5 * a
    ymin = ym - 0.5 * a

    roi_filter = (
        (xyz_original[:, 0] < xmax)
        & (xyz_original[:, 0] > xmin)
        & (xyz_original[:, 1] < ymax)
        & (xyz_original[:, 1] > ymin)
    )

    xyz_roi = xyz_original[roi_filter]
    if xyz_roi.shape[0] == 0:
        print("ROI is empty. Skipping.")
        return None

    xyz_proc = xyz_roi.copy()
    xyz_proc[:, 2] = -1 * xyz_proc[:, 2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_proc)

    distance_threshold = ransac_params.get("distance_thresh", 0.01)
    ransac_n = ransac_params.get("ransac_n", 3)
    num_iterations = ransac_params.get("num_iter", 1000)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    mask = np.ones(xyz_proc.shape[0], dtype=bool)
    if len(inliers) > 0:
        mask[inliers] = False

    if not np.any(mask):
        print(
            "RANSAC did not identify any outliers (potential groove points). Skipping."
        )
        return None

    depth, radius, _ = damage_calculation(xyz_proc, mask)
    return depth, radius


def evaluate():
    points3d_file = "pycalib/data/cache/hole10.npy"
    xyz_original = np.load(points3d_file)

    interpolate(xyz_original, vis=False)

    num_evals = 1
    all_depths = []
    all_radii = []

    roi_parameters = {"xm": 0.2, "ym": 0.5, "a": 5}
    ransac_parameters = {
        "distance_thresh": 0.01,
        "ransac_n": 3,
        "num_iter": 1000,
    }

    for i in range(num_evals):
        result = process_roi(xyz_original, roi_parameters, ransac_parameters)

        if result is not None:
            depth, radius = result
            all_depths.append(depth)
            all_radii.append(radius)
        else:
            print(f"Evaluation {i + 1}: Failed or skipped.")

    if all_depths:
        valid_depths = np.array(all_depths)
        valid_radii = np.array(all_radii)
        print(f"Mean Depth: {np.mean(valid_depths):.4f}")
        print(f"Std Dev Depth: {np.std(valid_depths):.4f}")
        print(f"Mean Radius: {np.mean(valid_radii):.4f}")
        print(f"Std Dev Radius: {np.std(valid_radii):.4f}")
    else:
        print("\nNo valid results obtained from any evaluation.")


if __name__ == "__main__":
    evaluate()
