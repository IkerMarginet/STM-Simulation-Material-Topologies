import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union

# Configuration dictionary for default lattice and STM parameters
CONFIG = {
    "si_111": {"a": 0.5431 / np.sqrt(2)},
    "graphene": {"a": 0.142},
    "graphite": {"a": 0.142, "c": 0.335, "nlayers": 3},
    "au_111": {"a": 0.4078 / np.sqrt(2)},
    "scan_size": (5.0, 5.0),
    "nx": 30,
    "ny": 30,
    "stm": {
        "resolution": 0.02,
        "tip_height": 0.35,
        "decay_const": 10.0,
        "noise_level": 0.05,
        "atom_height": 0.0
    },
    "defects": {
        "vacancy_prob": 0.03,
        "impurity_prob": 0.05,
        "displacement_scale": 0.05
    }
}

class Lattice:
    """Class to generate atomic positions for various lattice types."""
    
    def __init__(self, lattice_type: str, nx: int, ny: int, params: Dict) -> None:
        """
        Initialize lattice with type and dimensions.

        Args:
            lattice_type (str): Type of lattice ("si_111", "graphene", "graphite", "au_111").
            nx (int): Number of unit cells along x-direction.
            ny (int): Number of unit cells along y-direction.
            params (dict): Lattice parameters (e.g., {"a": 0.142, "c": 0.335}).

        Raises:
            ValueError: If lattice_type is unknown or parameters are invalid.
        """
        self.lattice_type = lattice_type.lower()
        self.nx = self._validate_positive_int(nx, "nx")
        self.ny = self._validate_positive_int(ny, "ny")
        self.params = params
        self.a1, self.a2, self.basis = self._set_lattice_vectors()

    def _validate_positive_int(self, value: int, name: str) -> int:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _set_lattice_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Define lattice vectors and basis for different lattice types."""
        if self.lattice_type == "si_111":
            a = self.params.get("a", 0.5431 / np.sqrt(2))
            if a <= 0:
                raise ValueError("Lattice constant 'a' must be positive")
            return (np.array([1, 0]) * a, 
                    np.array([0.5, np.sqrt(3)/2]) * a, 
                    np.array([[0, 0]]))
        elif self.lattice_type == "graphene":
            a = self.params.get("a", 0.142)
            if a <= 0:
                raise ValueError("Lattice constant 'a' must be positive")
            a_scaled = a * np.sqrt(3)
            return (np.array([1, 0]) * a_scaled,
                    np.array([0.5, np.sqrt(3)/2]) * a_scaled,
                    np.array([[0, 0], [a, 0]]))
        elif self.lattice_type == "graphite":
            a = self.params.get("a", 0.142)
            c = self.params.get("c", 0.335)
            if a <= 0 or c <= 0:
                raise ValueError("Lattice constants 'a' and 'c' must be positive")
            a_scaled = a * np.sqrt(3)
            return (np.array([1, 0]) * a_scaled,
                    np.array([0.5, np.sqrt(3)/2]) * a_scaled,
                    np.array([[0, 0], [a, 0]]))
        elif self.lattice_type == "au_111":
            a = self.params.get("a", 0.4078 / np.sqrt(2))
            if a <= 0:
                raise ValueError("Lattice constant 'a' must be positive")
            return (np.array([1, 0]) * a,
                    np.array([0.5, np.sqrt(3)/2]) * a,
                    np.array([[0, 0]]))
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")

    def generate(self, nlayers: int = 1) -> np.ndarray:
        """
        Generate lattice positions.

        Args:
            nlayers (int, optional): Number of layers (for graphite). Defaults to 1.

        Returns:
            np.ndarray: Array of shape (N, 2) with (x, y) positions in nm.
        """
        positions = []
        for layer in range(nlayers if self.lattice_type == "graphite" else 1):
            shift = ([self.params["a"]/3, self.params["a"]/(3*np.sqrt(3))]
                     if self.lattice_type == "graphite" and layer % 2 else [0, 0])
            ix = np.arange(self.nx)
            iy = np.arange(self.ny)
            xx, yy = np.meshgrid(ix, iy, indexing='ij')
            lattice_points = xx[..., None] * self.a1 + yy[..., None] * self.a2 + shift
            layer_pos = lattice_points.reshape(-1, 2)[:, None, :] + self.basis
            positions.append(layer_pos.reshape(-1, 2))
        return np.vstack(positions)

def apply_strain(positions: np.ndarray, strain_tensor: np.ndarray) -> np.ndarray:
    """
    Apply linear strain tensor (2x2) to all positions.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with (x, y) positions.
        strain_tensor (np.ndarray): 2x2 strain tensor (e.g., [[1+e, 0], [0, 1-e]]).

    Returns:
        np.ndarray: Transformed positions.

    Raises:
        ValueError: If strain_tensor is not 2x2 or positions is empty.
    """
    if strain_tensor.shape != (2, 2):
        raise ValueError("strain_tensor must be a 2x2 matrix")
    if positions.size == 0:
        raise ValueError("No positions provided")
    return positions @ strain_tensor.T

def introduce_vacancies(positions: np.ndarray, vacancy_prob: float = 0.01) -> np.ndarray:
    """
    Randomly remove atoms to simulate vacancies.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with (x, y) positions.
        vacancy_prob (float, optional): Probability of removing an atom. Defaults to 0.01.

    Returns:
        np.ndarray: Positions after removing vacancies.

    Raises:
        ValueError: If vacancy_prob is not in [0, 1] or positions is empty.
    """
    if not 0 <= vacancy_prob <= 1:
        raise ValueError("vacancy_prob must be between 0 and 1")
    if positions.size == 0:
        raise ValueError("No positions provided")
    mask = np.random.rand(len(positions)) > vacancy_prob
    return positions[mask]

def introduce_impurities(positions: np.ndarray, impurity_prob: float = 0.01,
                        displacement_scale: float = 0.05) -> np.ndarray:
    """
    Randomly displace atoms to simulate impurities.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with (x, y) positions.
        impurity_prob (float, optional): Probability of displacing an atom. Defaults to 0.01.
        displacement_scale (float, optional): Scale of displacements in nm. Defaults to 0.05.

    Returns:
        np.ndarray: Positions with displaced atoms.

    Raises:
        ValueError: If impurity_prob is not in [0, 1], displacement_scale <= 0, or positions is empty.
    """
    if not 0 <= impurity_prob <= 1:
        raise ValueError("impurity_prob must be between 0 and 1")
    if displacement_scale <= 0:
        raise ValueError("displacement_scale must be positive")
    if positions.size == 0:
        raise ValueError("No positions provided")
    positions_new = positions.copy()
    mask = np.random.rand(len(positions)) < impurity_prob
    displacements = np.random.randn(np.sum(mask), 2) * displacement_scale
    positions_new[mask] += displacements
    return positions_new

def simulate_stm_image(positions: np.ndarray, scan_size: Tuple[float, float] = (10, 10),
                      resolution: float = 0.05, atom_height: float = 0.0,
                      tip_height: float = 0.3, decay_const: float = 10.0,
                      noise_level: float = 0.05, chunk_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate STM image as a 2D array with tunneling contributions.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with (x, y) positions in nm.
        scan_size (tuple): (width_x, width_y) of simulated area in nm.
        resolution (float): nm per pixel.
        atom_height (float): Atom plane z in nm.
        tip_height (float): Tip z above plane in nm.
        decay_const (float): Decay constant for tunneling in 1/nm.
        noise_level (float): Relative noise amplitude in STM signal.
        chunk_size (int): Number of atoms to process per chunk for memory efficiency.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (x_grid, y_grid, image) where image is the STM signal.

    Raises:
        ValueError: If inputs are invalid (e.g., negative resolution, empty positions).
    """
    if positions.size == 0:
        raise ValueError("No positions provided")
    if resolution <= 0 or scan_size[0] <= 0 or scan_size[1] <= 0:
        raise ValueError("scan_size and resolution must be positive")
    if tip_height <= 0 or decay_const <= 0:
        raise ValueError("tip_height and decay_const must be positive")

    # Filter atoms within scan area plus buffer
    buffer = 3 / decay_const  # Distance where exp(-decay_const * r) is negligible
    mask = ((positions[:, 0] >= -buffer) & (positions[:, 0] <= scan_size[0] + buffer) &
            (positions[:, 1] >= -buffer) & (positions[:, 1] <= scan_size[1] + buffer))
    positions = positions[mask]
    
    xpts, ypts = int(scan_size[0] / resolution), int(scan_size[1] / resolution)
    x = np.linspace(0, scan_size[0], xpts)
    y = np.linspace(0, scan_size[1], ypts)
    xx, yy = np.meshgrid(x, y)
    img = np.zeros((xpts, ypts))

    for i in range(0, len(positions), chunk_size):
        chunk = positions[i:i + chunk_size]
        dx = xx[None, :, :] - chunk[:, 0][:, None, None]
        dy = yy[None, :, :] - chunk[:, 1][:, None, None]
        dz = np.sqrt(dx**2 + dy**2) + tip_height
        img += np.exp(-decay_const * dz).sum(axis=0)

    max_img = np.max(img)
    if max_img > 0:
        img /= max_img
    img += noise_level * np.random.randn(*img.shape)
    img = np.clip(img, 0, 1)
    return xx, yy, img

def plot_lattice(positions: np.ndarray, scan_size: Tuple[float, float], title: str = "Atomic Lattice",
                 color: str = 'r', ax: Optional[plt.Axes] = None,
                 compare: Optional[Dict] = None) -> Optional[plt.Axes]:
    """
    Plot lattice positions, optionally with a comparison lattice.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with (x, y) positions.
        scan_size (tuple): (width_x, width_y) in nm.
        title (str): Plot title.
        color (str): Color of scatter points.
        ax (plt.Axes, optional): Matplotlib axes to plot on.
        compare (dict, optional): Dictionary with 'positions', 'title', and 'color' for comparison plot.

    Returns:
        plt.Axes or None: Axes if single plot, None if comparison plot.

    Raises:
        ValueError: If positions is empty.
    """
    if positions.size == 0:
        raise ValueError("No positions provided")
    
    # Single plot case
    if compare is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(positions[:, 0], positions[:, 1], s=10, c=color, label=title)
        ax.set_xlim(0, scan_size[0])
        ax.set_ylim(0, scan_size[1])
        ax.set_aspect('equal')
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(title)
        ax.legend()
        return ax
    
    # Comparison plot case
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first lattice
    ax1.scatter(positions[:, 0], positions[:, 1], s=10, c=color, label=title)
    ax1.set_xlim(0, scan_size[0])
    ax1.set_ylim(0, scan_size[1])
    ax1.set_aspect('equal')
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title(title)
    ax1.legend()
    
    # Plot second (comparison) lattice
    ax2.scatter(compare['positions'][:, 0], compare['positions'][:, 1], s=10,
                c=compare.get('color', 'b'), label=compare.get('title', 'Comparison'))
    ax2.set_xlim(0, scan_size[0])
    ax2.set_ylim(0, scan_size[1])
    ax2.set_aspect('equal')
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")
    ax2.set_title(compare.get('title', 'Comparison'))
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    return None

def plot_stm_comparison(x_grid1: np.ndarray, y_grid1: np.ndarray, img1: np.ndarray,
                        x_grid2: np.ndarray, y_grid2: np.ndarray, img2: np.ndarray,
                        label1: str = "Perfect", label2: str = "With Defects") -> None:
    """
    Plot two STM images side by side.

    Args:
        x_grid1, y_grid1 (np.ndarray): Grid coordinates for first image.
        img1 (np.ndarray): First STM image.
        x_grid2, y_grid2 (np.ndarray): Grid coordinates for second image.
        img2 (np.ndarray): Second STM image.
        label1, label2 (str): Titles for the images.

    Raises:
        ValueError: If image shapes or grids are invalid.
    """
    if img1.shape != img2.shape or x_grid1.shape != x_grid2.shape:
        raise ValueError("Inconsistent image or grid shapes")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    c1 = ax1.pcolormesh(x_grid1, y_grid1, img1, shading="auto", cmap='hot', vmin=0, vmax=1)
    fig.colorbar(c1, ax=ax1, label="STM Signal (a.u.)")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title(f"STM: {label1}")
    
    c2 = ax2.pcolormesh(x_grid2, y_grid2, img2, shading="auto", cmap='hot', vmin=0, vmax=1)
    fig.colorbar(c2, ax=ax2, label="STM Signal (a.u.)")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")
    ax2.set_title(f"STM: {label2}")
    
    plt.tight_layout()
    plt.show()

def generate_lattice(surface: str, config: Dict) -> np.ndarray:
    """
    Generate lattice positions for a given surface type.

    Args:
        surface (str): Lattice type ("si_111", "graphene", "graphite", "au_111").
        config (dict): Configuration dictionary with lattice and scan parameters.

    Returns:
        np.ndarray: Clipped positions within scan_size.

    Raises:
        ValueError: If surface is unknown or scan_size is invalid.
    """
    if surface.lower() not in CONFIG:
        raise ValueError(f"Unknown surface: {surface}")
    nlayers = config.get("graphite", {}).get("nlayers", 1) if surface.lower() == "graphite" else 1
    lattice = Lattice(surface, config["nx"], config["ny"], config[surface.lower()])
    positions = lattice.generate(nlayers=nlayers)
    
    scan_size = config["scan_size"]
    mask = ((positions[:, 0] >= 0) & (positions[:, 0] <= scan_size[0]) &
            (positions[:, 1] >= 0) & (positions[:, 1] <= scan_size[1]))
    return positions[mask]

def get_float_input(prompt: str, default: float, min_val: float = 0.0) -> float:
    """
    Prompt for a float input with validation.

    Args:
        prompt (str): Input prompt message.
        default (float): Default value if input is empty.
        min_val (float): Minimum allowed value.

    Returns:
        float: Validated float value.
    """
    while True:
        try:
            value = input(f"{prompt} (default: {default}): ").strip()
            if value == "":
                return default
            value = float(value)
            if value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def get_int_input(prompt: str, default: int, min_val: int = 1) -> int:
    """
    Prompt for an integer input with validation.

    Args:
        prompt (str): Input prompt message.
        default (int): Default value if input is empty.
        min_val (int): Minimum allowed value.

    Returns:
        int: Validated integer value.
    """
    while True:
        try:
            value = input(f"{prompt} (default: {default}): ").strip()
            if value == "":
                return default
            value = int(value)
            if value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer")

def main_menu() -> Dict:
    """
    Display a menu to collect user inputs for simulation parameters.

    Returns:
        dict: Configuration dictionary with user-specified parameters.
    """
    print("\n=== Lattice and STM Simulation Menu ===")
    print("Available lattice types: si_111, graphene, graphite, au_111")
    
    # Lattice type
    valid_lattices = ["si_111", "graphene", "graphite", "au_111"]
    while True:
        surface = input("Select lattice type (si_111, graphene, graphite, au_111): ").strip().lower()
        if surface in valid_lattices:
            break
        print("Invalid lattice type. Choose from: si_111, graphene, graphite, au_111")
    
    # Lattice dimensions
    nx = get_int_input("Number of unit cells in x-direction (nx)", CONFIG["nx"])
    ny = get_int_input("Number of unit cells in y-direction (ny)", CONFIG["ny"])
    
    # Scan size
    scan_x = get_float_input("Scan size x (nm)", CONFIG["scan_size"][0])
    scan_y = get_float_input("Scan size y (nm)", CONFIG["scan_size"][1])
    
    # Defect parameters
    vacancy_prob = get_float_input("Vacancy probability (0 to 1)", CONFIG["defects"]["vacancy_prob"], min_val=0.0)
    impurity_prob = get_float_input("Impurity probability (0 to 1)", CONFIG["defects"]["impurity_prob"], min_val=0.0)
    displacement_scale = get_float_input("Displacement scale for impurities (nm)", 
                                        CONFIG["defects"]["displacement_scale"])
    
    # STM parameters
    resolution = get_float_input("STM resolution (nm/pixel)", CONFIG["stm"]["resolution"])
    tip_height = get_float_input("STM tip height (nm)", CONFIG["stm"]["tip_height"])
    noise_level = get_float_input("STM noise level", CONFIG["stm"]["noise_level"], min_val=0.0)
    
    # Graphite-specific parameter
    nlayers = 1
    if surface == "graphite":
        nlayers = get_int_input("Number of graphite layers", CONFIG["graphite"]["nlayers"])
    
    # Create configuration
    user_config = CONFIG.copy()
    user_config["nx"] = nx
    user_config["ny"] = ny
    user_config["scan_size"] = (scan_x, scan_y)
    user_config["defects"] = {
        "vacancy_prob": vacancy_prob,
        "impurity_prob": impurity_prob,
        "displacement_scale": displacement_scale
    }
    user_config["stm"] = CONFIG["stm"].copy()
    user_config["stm"].update({
        "resolution": resolution,
        "tip_height": tip_height,
        "noise_level": noise_level
    })
    if surface == "graphite":
        user_config["graphite"]["nlayers"] = nlayers
    
    return surface, user_config

def main() -> None:
    """Main function to generate and visualize lattice and STM images."""
    np.random.seed(1)
    
    # Get user inputs via menu
    surface, user_config = main_menu()
    material_name = surface.capitalize()

    # Generate perfect and defective lattices
    perfect_positions = generate_lattice(surface, user_config)
    defective_positions = introduce_vacancies(perfect_positions, 
                                            vacancy_prob=user_config["defects"]["vacancy_prob"])
    defective_positions = introduce_impurities(defective_positions, 
                                             impurity_prob=user_config["defects"]["impurity_prob"],
                                             displacement_scale=user_config["defects"]["displacement_scale"])

    # Simulate STM images
    x_grid1, y_grid1, img1 = simulate_stm_image(perfect_positions, **user_config["stm"], 
                                               scan_size=user_config["scan_size"])
    x_grid2, y_grid2, img2 = simulate_stm_image(defective_positions, **user_config["stm"], 
                                               scan_size=user_config["scan_size"])

    # Plot results
    plot_lattice(perfect_positions, user_config["scan_size"], title=f"Perfect {material_name}",
                 color='r', compare={'positions': defective_positions, 
                                    'title': f"{material_name} with Defects", 'color': 'b'})
    plot_stm_comparison(x_grid1, y_grid1, img1, x_grid2, y_grid2, img2,
                       label1=f"Perfect {material_name}", label2=f"{material_name} with Defects")

if __name__ == "__main__":
    main()