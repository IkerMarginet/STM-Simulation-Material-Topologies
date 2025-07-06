import numpy as np
import matplotlib.pyplot as plt

#####################################################
# Lattice model functions
#####################################################

def lattice_si_111(nx, ny, a=0.5431/np.sqrt(2)):
    """
    Generates Si(111) surface lattice (hexagonal), single layer.
    a = 0.5431/np.sqrt(2) nm (adjusted from bulk lattice constant of Si).
    """
    a1 = a * np.array([1, 0])
    a2 = a * np.array([0.5, np.sqrt(3)/2])
    ix = np.arange(nx)
    iy = np.arange(ny)
    xx, yy = np.meshgrid(ix, iy, indexing='ij')
    positions = xx[..., None]*a1 + yy[..., None]*a2
    return positions.reshape(-1, 2)

def lattice_au_111(nx, ny, a=0.4078/np.sqrt(2)):
    """
    Generates Au(111) surface lattice (hexagonal), single layer.
    a = 0.4078/np.sqrt(2) nm (adjusted from bulk lattice constant of Au).
    """
    a1 = a * np.array([1, 0])
    a2 = a * np.array([0.5, np.sqrt(3)/2])
    ix = np.arange(nx)
    iy = np.arange(ny)
    xx, yy = np.meshgrid(ix, iy, indexing='ij')
    positions = xx[..., None]*a1 + yy[..., None]*a2
    return positions.reshape(-1, 2)

def lattice_graphene(nx, ny, a=0.142):
    """
    Graphene: honeycomb lattice (2 atoms/unit cell).
    a = 0.142 nm (C-C bond length in graphene).
    """
    a1 = a * np.sqrt(3) * np.array([1, 0])
    a2 = a * np.sqrt(3) * np.array([0.5, np.sqrt(3)/2])
    basis = np.array([[0, 0], [a, 0]])
    ix = np.arange(nx)
    iy = np.arange(ny)
    xx, yy = np.meshgrid(ix, iy, indexing='ij')
    lattice_points = xx[..., None]*a1 + yy[..., None]*a2  # shape (nx, ny, 2)
    lattice_points = lattice_points.reshape(-1, 2)
    positions = lattice_points[:, None, :] + basis  # shape (nx*ny, 2, 2)
    return positions.reshape(-1, 2)

#####################################################
# Nonperiodic Features: Strain / Defects
#####################################################

def apply_strain(positions, strain_tensor):
    """
    Apply linear strain tensor (2x2) to all positions.
    E.g., [[1+e,0],[0,1-e]] for uniaxial strain.
    """
    return positions @ strain_tensor.T

def introduce_vacancies(positions, vacancy_prob=0.01):
    """
    Randomly remove some atoms to simulate vacancies/defects.
    """
    mask = np.random.rand(len(positions)) > vacancy_prob
    return positions[mask]

def introduce_impurities(positions, impurity_prob=0.01, displacement_scale=0.05):
    """
    Randomly displace some atoms ("impurities") by small random displacements.
    displacement_scale is in nm.
    """
    positions_new = positions.copy()
    mask = np.random.rand(len(positions)) < impurity_prob
    displacements = np.random.randn(np.sum(mask), 2) * displacement_scale
    positions_new[mask] += displacements
    return positions_new

#####################################################
# STM "Tunneling Current/Intensity" Calculation
#####################################################

def simulate_stm_image(positions, scan_size=(10, 10), resolution=0.05,
                       atom_height=0.0, tip_height=0.3, decay_const=10.0, noise_level=0.05):
    """
    Simulates STM image as 2D array with realistic tunneling contributions.
    'positions' : array of atomic (x, y) positions.
    scan_size : (width_x, width_y) of simulated area in nm.
    resolution : nm per pixel.
    atom_height : atom plane z [nm].
    tip_height : tip z above plane [nm].
    decay_const : decay constant for tunneling [1/nm].
    noise_level : relative noise amplitude in STM signal.
    
    Returns:
      xgrid, ygrid, img (STM image as 2D array)
    """
    xpts, ypts = int(scan_size[0] / resolution), int(scan_size[1] / resolution)
    x = np.linspace(0, scan_size[0], xpts)
    y = np.linspace(0, scan_size[1], ypts)
    xx, yy = np.meshgrid(x, y)

    # Vectorized distance calculation: shape (num_atoms, ypts, xpts)
    # Use broadcasting carefully to avoid huge memory if very large lattices
    dx = xx[None, :, :] - positions[:, 0][:, None, None]
    dy = yy[None, :, :] - positions[:, 1][:, None, None]
    dz = np.sqrt(dx**2 + dy**2) + tip_height

    # Calculate tunneling current contribution from each atom
    currents = np.exp(-decay_const * dz)  # shape (num_atoms, ypts, xpts)
    img = currents.sum(axis=0)  # Sum over atoms

    max_img = np.max(img)
    if max_img > 0:
        img /= max_img
    img += noise_level * np.random.randn(*img.shape)
    img = np.clip(img, 0, 1)
    return xx, yy, img

#####################################################
# Visualization functions
#####################################################

def plot_lattice(positions, scan_size, atom_color='r', ax=None):
    s = 10  # marker size
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=s, c=atom_color)
    ax.set_xlim(0, scan_size[0])
    ax.set_ylim(0, scan_size[1])
    ax.set_aspect('equal')
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("Atomic Lattice Configuration")
    return ax

def plot_stm(xx, yy, img):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(xx, yy, img, shading="auto", cmap='hot', vmin=0, vmax=1)
    fig.colorbar(c, ax=ax, label="STM Signal (arbitrary units)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("Simulated STM Image")
    plt.show()

def plot_comparison(positions1, positions2, scan_size, label1="Perfect", label2="With Impurities"):
    """
    Plot two lattices side by side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot perfect lattice
    ax1.scatter(positions1[:, 0], positions1[:, 1], s=10, c='b', label=label1)
    ax1.set_xlim(0, scan_size[0])
    ax1.set_ylim(0, scan_size[1])
    ax1.set_aspect('equal')
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title(f"{label1} Si(111) Lattice")
    ax1.legend()
    
    # Plot lattice with impurities
    ax2.scatter(positions2[:, 0], positions2[:, 1], s=10, c='r', label=label2)
    ax2.set_xlim(0, scan_size[0])
    ax2.set_ylim(0, scan_size[1])
    ax2.set_aspect('equal')
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")
    ax2.set_title(f"{label2} Si(111) Lattice")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_stm_comparison(xx1, yy1, img1, xx2, yy2, img2, label1="Perfect", label2="With Impurities"):
    """
    Plot two STM images side by side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot STM image for perfect lattice
    c1 = ax1.pcolormesh(xx1, yy1, img1, shading="auto", cmap='hot', vmin=0, vmax=1)
    fig.colorbar(c1, ax=ax1, label="STM Signal (a.u.)")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title(f"STM: {label1} Si(111)")
    
    # Plot STM image for lattice with impurities
    c2 = ax2.pcolormesh(xx2, yy2, img2, shading="auto", cmap='hot', vmin=0, vmax=1)
    fig.colorbar(c2, ax=ax2, label="STM Signal (a.u.)")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")
    ax2.set_title(f"STM: {label2} Si(111)")
    
    plt.tight_layout()
    plt.show()

#####################################################
# Main script
#####################################################

def main():
    nx, ny = 30, 30  # lattice repetitions
    scan_size = (5, 5)  # nm (scan area)
    res = 0.02  # nm per pixel (resolution)

    # 1. Generate perfect Si(111) lattice
    lattice_type = 'Si(111)'
    positions_perfect = lattice_si_111(nx, ny)

    # 2. Generate Si(111) lattice with impurities and vacancies
    positions_impure = positions_perfect.copy()
    # Introduce vacancies
    positions_impure = introduce_vacancies(positions_impure, vacancy_prob=0.03)
    # Introduce impurities (displaced atoms)
    positions_impure = introduce_impurities(positions_impure, impurity_prob=0.05, displacement_scale=0.05)

    # 3. Filter atoms within scan region for both lattices
    inside_perfect = ((positions_perfect[:, 0] >= 0) & (positions_perfect[:, 0] <= scan_size[0]) &
                      (positions_perfect[:, 1] >= 0) & (positions_perfect[:, 1] <= scan_size[1]))
    positions_perfect = positions_perfect[inside_perfect]

    inside_impure = ((positions_impure[:, 0] >= 0) & (positions_impure[:, 0] <= scan_size[0]) &
                     (positions_impure[:, 1] >= 0) & (positions_impure[:, 1] <= scan_size[1]))
    positions_impure = positions_impure[inside_impure]

    # 4. Simulate STM images for both lattices
    xx1, yy1, img1 = simulate_stm_image(positions_perfect, scan_size=scan_size, resolution=res)
    xx2, yy2, img2 = simulate_stm_image(positions_impure, scan_size=scan_size, resolution=res)

    # 5. Plot atomic lattices
    plot_comparison(positions_perfect, positions_impure, scan_size, label1="Perfect", label2="Impurities")

    # 6. Plot STM images
    plot_stm_comparison(xx1, yy1, img1, xx2, yy2, img2, label1="Perfect", label2="Impurities")

if __name__ == "__main__":
    main()
