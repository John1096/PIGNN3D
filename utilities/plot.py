"""Export rollouts as VTK/Excel data or comparison animations."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

def save_simulation_to_vtk(z_net, data_list, output_dir, prefix="simulation_data", input_number=None):
    """
    Save to VTK (legacy .vtk PolyData), one file per time step, under:
        {output_dir}/vtk_output/{input_number}/

    Args:
        z_net (torch.Tensor): [T, N, 5] -> (vx, vy, vz, pressure, temperature)
        data_list (list): list of PyG Data; data_list[0].q_0 is [N, 3] coords (X,Y,Z)
        output_dir (str): base output directory
        prefix (str): filename prefix
        input_number (int|str): e.g., 95 (will create folder named "95")
    """
    if input_number is None:
        raise ValueError("Please provide input_number (e.g., args.cooling).")

    # Always write under {output_dir}/vtk_output/{input_number}
    target_dir = os.path.join(output_dir, "vtk_output" + str(input_number))
    os.makedirs(target_dir, exist_ok=True)

    # Coordinates
    q0 = data_list[0].q_0
    if isinstance(q0, torch.Tensor):
        q0 = q0.detach().cpu().numpy()
    else:
        q0 = np.asarray(q0)
    assert q0.ndim == 2 and q0.shape[1] == 3, "q_0 must be [N,3] (X,Y,Z)."

    # Simulation outputs
    if isinstance(z_net, torch.Tensor):
        z_net = z_net.detach().cpu().numpy()
    else:
        z_net = np.asarray(z_net)
    assert z_net.ndim == 3 and z_net.shape[2] == 5, "z_net must be [T,N,5]."

    T, N, _ = z_net.shape
    assert q0.shape[0] == N, "Point count mismatch between q_0 and z_net."

    written = []
    for t in range(T):
        vx = z_net[t, :, 0]
        vy = z_net[t, :, 1]
        vz = z_net[t, :, 2]
        pr = z_net[t, :, 3]
        tm = z_net[t, :, 4]

        path = os.path.join(target_dir, f"{prefix}_time_{t:03d}.vtk")
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{prefix} timestep {t}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            f.write(f"POINTS {N} float\n")
            for i in range(N):
                x, y, z = q0[i, 0], q0[i, 1], q0[i, 2]
                f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

            f.write(f"VERTICES {N} {2*N}\n")
            for i in range(N):
                f.write(f"1 {i}\n")

            f.write(f"POINT_DATA {N}\n")

            f.write("VECTORS velocity float\n")
            for i in range(N):
                f.write(f"{vx[i]:.8e} {vy[i]:.8e} {vz[i]:.8e}\n")

            f.write("SCALARS pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(N):
                f.write(f"{pr[i]:.8e}\n")

            f.write("SCALARS temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(N):
                f.write(f"{tm[i]:.8e}\n")

        written.append(path)

    print(f"VTK written under: {target_dir}")
    return written


def save_simulation_data(z_net, data_list, output_dir, prefix="simulation_data"):
    """
    Save Q_0 (X, Y, Z coordinates) and z_net (velocity_x, velocity_y, velocity_z, pressure, temperature)
    into separate Excel files for each time step.

    Parameters:
        z_net (torch.Tensor): Tensor containing velocity_x, velocity_y, velocity_z, pressure, temperature for each time step.
        data_list (list): List containing dataset objects, where q_0 represents (X, Y, Z) coordinates.
        output_dir (str): Directory to save the Excel files.
        prefix (str): Prefix for the Excel filenames.
    """
    T = z_net.size(0)  # Number of time steps
    q_0 = data_list[0].q_0.cpu().numpy()  # Extract (X, Y, Z) coordinates
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Loop through each time step and save data separately
    for t in range(T):
        z_net_np = z_net[t].cpu().numpy()  # Convert z_net for timestep t to NumPy array

        # Create a DataFrame
        df = pd.DataFrame({
            'X': q_0[:, 0],
            'Y': q_0[:, 1],
            'Z': q_0[:, 2],
            'Velocity_X': z_net_np[:, 0],
            'Velocity_Y': z_net_np[:, 1],
            'Velocity_Z': z_net_np[:, 2],
            'Pressure': z_net_np[:, 3],
            'Temperature': z_net_np[:, 4],
        })

        # Define the file name for this time step
        file_name = os.path.join(output_dir, f"{prefix}_time_{t:03d}.xlsx")

        # Save DataFrame to an Excel file
        df.to_excel(file_name, index=False)

    return output_dir

def plot_3D(z_net, z_gt, data_list, output_dir, var=6):
    T = z_net.size(0)
    n = data_list[0].n

    # Plot initialization
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

    # Adjust ranges
    X, Y, Z = z_gt[:,:,0].numpy(), z_gt[:,:,1].numpy(), z_gt[:,:,2].numpy()
    z_min, z_max = z_gt[:,:,var].min(), z_gt[:,:,var].max()
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[0,:,0], z_net[0,:,1], z_net[0,:,2]
    q1_gt, q2_gt, q3_gt = z_gt[0,:,0], z_gt[0,:,1], z_gt[0,:,2]
    var_net, var_gt = z_net[0,:,var], z_gt[0,:,var]
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [yb], [zb], 'w')
        ax2.plot([xb], [yb], [zb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_net[n[:,0]==1], vmax=z_max, vmin=z_min)
    ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
    s2 = ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gt[n[:,0]==1], vmax=z_max, vmin=z_min)
    ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
            ax2.plot([xb], [yb], [zb], 'w')
        # Scatter points
        q1_net, q2_net, q3_net = z_net[snap,:,0], z_net[snap,:,1], z_net[snap,:,2]
        q1_gt, q2_gt, q3_gt = z_gt[snap,:,0], z_gt[snap,:,1], z_gt[snap,:,2]
        var_net, var_gt = z_net[snap,:,var], z_gt[snap,:,var]
        ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_net[n[:,0]==1], vmax=z_max, vmin=z_min)
        ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
        ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gt[n[:,0]==1], vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')

        return fig,
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    save_dir = os.path.join(output_dir, 'beam.gif')
    anim.save(save_dir, writer=writergif)


def plot_2D(z_net, z_gt, data_list, output_dir, var=0):
    T = z_net.size(0)
    res = 100
    q_0 = data_list[0].q_0
    n = data_list[0].n

    # Plot initialization
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')

    # Adjust ranges
    X, Y = q_0[:,0].cpu().numpy(), q_0[:,1].cpu().numpy()
    z_min, z_max = z_gt[:,:,var].min(), z_gt[:,:,var].max()
    levels = np.linspace(z_min, z_max, 10, endpoint = True)
    ax1.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax2.axis([X.min(), X.max(), Y.min(), Y.max()])

    # Interpolate data to grid
    zi_net = np.zeros([T, res, res])
    zi_gt = np.zeros([T, res, res])
    xi = np.linspace(X.min(), X.max(), res)
    yi = np.linspace(Y.min(), Y.max(), res)
    for t in range(T):
        Z_gt = z_gt[t,:,var].numpy()
        Z_net = z_net[t,:,var].numpy()
        zi_net[t] = griddata((X, Y), Z_net, (xi[None,:], yi[:,None]), method='cubic')
        zi_gt[t] = griddata((X, Y), Z_gt, (xi[None,:], yi[:,None]), method='cubic')

    # Initial snapshot
    c1 = ax1.contourf(xi, yi, zi_net[0], levels=levels, vmax=z_max, vmin=z_min)
    c2 = ax2.contourf(xi, yi, zi_gt[0], levels=levels, vmax=z_max, vmin=z_min)
    # Plot Boundaries
    ax1.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
    ax2.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    # Colorbar
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c1, ax=ax1, cax=cax1, ticks=np.linspace(z_min, z_max, 5), format='%.2f')
    fig.colorbar(c2, ax=ax2, cax=cax2, ticks=np.linspace(z_min, z_max, 5), format='%.2f')

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        # Boundaries
        ax1.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
        ax2.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        # Contour
        ax1.contourf(xi, yi, zi_net[snap], levels=levels, vmax=z_max, vmin=z_min)
        ax2.contourf(xi, yi, zi_gt[snap], levels=levels, vmax=z_max, vmin=z_min)
        return fig,
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=30)

    # Save as gif
    save_dir = os.path.join(output_dir, 'cylinder.gif')
    anim.save(save_dir, writer=writergif)


def plot_3DT(z_net, z_gt, data_list, output_dir, var=0):
    T = z_net.size(0)  # Number of time steps
    res = 50  # Grid resolution (smaller to optimize rendering)

    q_0 = data_list[0].q_0.cpu().numpy()  # Extract point positions (X, Y, Z)
    n = data_list[0].n.cpu().numpy()

    X, Y, Z = q_0[:,0], q_0[:,1], q_0[:,2]  # Extract coordinates

    # Set value ranges
    z_min, z_max = z_gt[:,:,var].min(), z_gt[:,:,var].max()

    # Generate grid for interpolation
    xi = np.linspace(X.min(), X.max(), res)
    yi = np.linspace(Y.min(), Y.max(), res)
    zi = np.linspace(Z.min(), Z.max(), res)
    xi, yi = np.meshgrid(xi, yi)  # Create 2D meshgrid (for slicing)

    # Initialize storage for interpolated data
    zi_net = np.zeros((T, res, res))
    zi_gt = np.zeros((T, res, res))

    # Interpolate data onto grid
    for t in range(T):
        Z_gt = z_gt[t, :, var].cpu().numpy()
        Z_net = z_net[t, :, var].cpu().numpy()

        # Interpolating only a 2D slice
        zi_net[t] = griddata((X, Y), Z_net, (xi, yi), method='cubic')
        zi_gt[t] = griddata((X, Y), Z_gt, (xi, yi), method='cubic')

    # Plot initialization
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    def animate(snap):
        ax1.cla()
        ax2.cla()

        ax1.set_title('Thermodynamics-informed GNN')
        ax2.set_title('Ground Truth')

        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

        # Set axis limits
        ax1.set_xlim([X.min(), X.max()])
        ax1.set_ylim([Y.min(), Y.max()])
        ax1.set_zlim([Z.min(), Z.max()])

        ax2.set_xlim([X.min(), X.max()])
        ax2.set_ylim([Y.min(), Y.max()])
        ax2.set_zlim([Z.min(), Z.max()])

        # Surface plot
        surf1 = ax1.plot_surface(xi, yi, zi_net[snap], cmap='viridis', vmin=z_min, vmax=z_max)
        surf2 = ax2.plot_surface(xi, yi, zi_gt[snap], cmap='viridis', vmin=z_min, vmax=z_max)

        return surf1, surf2

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=30)

    # Save animation as GIF
    save_dir = os.path.join(output_dir, 'cylinder_3D.gif')
    anim.save(save_dir, writer=writergif)

    plt.show()
