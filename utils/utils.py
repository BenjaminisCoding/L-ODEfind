import pandas as pd
import os
import re 
import matplotlib.pyplot as plt

from src.lib.simulation_manager import Oscilator, LorenzAttractor, RoselerAttractor
from src.scripts.ode.generate_data import generate_data, generate_data_with_noise


def print_discovered_equation(
    results_dir, 
    system_name, 
    obs_vars, 
    target_d, 
    max_poly, 
    params_id=0, 
    init_cond_id=0, 
    threshold=1e-5
):
    """
    Reads the L-ODEfind result file and prints the discovered differential equation.
    
    Args:
        results_dir (str): Path to the 'results' folder.
        system_name (str): Name of the system (e.g., 'LorenzAttractor').
        obs_vars (list): List of observed variables (e.g., ['x']).
        target_d (int): The order of the derivative that was targeted (dmax).
        max_poly (int): Maximum polynomial degree used.
        params_id (int): ID of the parameter set used (default 0).
        init_cond_id (int): ID of the initial condition used (default 0).
        threshold (float): Coefficients smaller than this magnitude are treated as zero.
    """
    
    # 1. Construct the folder path and filename based on the author's convention
    # Folder naming convention: {system}_{vars}_Odefind
    folder_name = f"{system_name}_{'_'.join(obs_vars)}_Odefind"
    full_dir = os.path.join(results_dir, folder_name)
    
    # File naming convention: 
    # solution-pdefind_coefs-vars_{vars}-dmax_{d}-poly_{p}-params_{p_id}-init_cond_{ic_id}-steps_0.csv
    vars_str = '_'.join(obs_vars)
    filename = (
        f"solution-pdefind_coefs-vars_{vars_str}-dmax_{target_d}-poly_{max_poly}-"
        f"params_{params_id}-init_cond_{init_cond_id}-steps_0.csv"
    )
    file_path = os.path.join(full_dir, filename)
    file_path = os.path.join(os.getcwd()[:-len(os.getcwd().split('/')[-1])-1], file_path)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Load the coefficients
    # The file is saved transposed: Rows = Library Features, Columns = Target Derivatives
    df = pd.read_csv(file_path, index_col=0)
    
    print(f"--- Discovered Equation for {system_name} (observed: {vars_str}) ---")
    
    # 3. Iterate through each target derivative (column)
    for target in df.columns:
        # Clean up the target name for display
        target_clean = clean_term_name(target)
        equation_terms = []
        
        # 4. Iterate through features (rows) to find active terms
        for feature, coeff in df[target].items():
            if abs(coeff) > threshold:
                # Clean up feature name
                feat_clean = clean_term_name(feature)
                
                # Format coefficient
                coeff_str = f"{coeff:.4f}"
                
                # Handle the constant term '1.0'
                if feat_clean == "1.0" or feat_clean == "1":
                    equation_terms.append(coeff_str)
                else:
                    equation_terms.append(f"({coeff_str} * {feat_clean})")
        
        # 5. Assemble and print
        if not equation_terms:
            rhs = "0"
        else:
            rhs = " + ".join(equation_terms)
            # Clean up "+ -" to just "- " for readability
            rhs = rhs.replace("+ -", "- ")
            
        print(f"{target_clean} = {rhs}")
        print("-" * 60)

def clean_term_name(term):
    """
    Helper to make internal variable names readable.
    Example: '1.0*Derivative(x(t), (t, 2))' -> 'd^2x/dt^2'
    """
    # Remove leading multipliers often added by sympy/pandas export
    if term.startswith("1.0*"):
        term = term[4:]
        
    # Replace Powers
    term = term.replace("**", "^")
    
    # Replace Derivative notations
    # Format: Derivative(x(t), t) -> dx/dt
    # Format: Derivative(x(t), (t, 2)) -> d^2x/dt^2
    if "Derivative" in term:
        # Extract variable name (e.g., x)
        var_part = term.split("(")[1].split("(")[0] # simple parse
        
        if ", (t," in term:
            # Higher order: Derivative(x(t), (t, 2))
            order = term.split(", (t,")[1].split(")")[0].strip()
            term = f"d^{order}{var_part}/dt^{order}"
        else:
            # First order: Derivative(x(t), t)
            term = f"d{var_part}/dt"
            
    # Clean up function notation x(t) -> x
    term = term.replace("(t)", "")
    
    return term


### New version 

def print_discovered_equation_v2(
    results_dir, 
    system_name, 
    obs_vars, 
    target_d, 
    max_poly, 
    params_id=0, 
    init_cond_id=0, 
    threshold=1e-5
):
    """
    Reads the L-ODEfind result file and prints the discovered differential equation
    with improved formatting and parsing.
    """
    
    # 1. Construct File Path
    folder_name = f"{system_name}_{'_'.join(obs_vars)}_Odefind"
    full_dir = os.path.join(results_dir, folder_name)
    
    vars_str = '_'.join(obs_vars)
    filename = (
        f"solution-pdefind_coefs-vars_{vars_str}-dmax_{target_d}-poly_{max_poly}-"
        f"params_{params_id}-init_cond_{init_cond_id}-steps_0.csv"
    )
    file_path = os.path.join(full_dir, filename)
    file_path = os.path.join(os.getcwd()[:-len(os.getcwd().split('/')[-1])-1], file_path)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print(f"--- Discovered Equation for {system_name} (observed: {vars_str}) ---")
    
    # 3. Parse Equations
    for target_col in df.columns:
        # Determine Left Hand Side (LHS) from the column header
        lhs = parse_term(target_col)
        
        rhs_terms = []
        
        # Iterate through features (rows)
        for feature_raw, coeff in df[target_col].items():
            if abs(coeff) > threshold:
                
                # Parse the feature string
                feature_clean = parse_term(feature_raw)
                
                # Format Coefficient
                is_negative = coeff < 0
                val = abs(coeff)
                val_str = f"{val:.4f}"
                
                # Construct Term String
                if feature_clean == "1" or feature_clean == "":
                    # It's a constant
                    term_str = val_str
                else:
                    # It's a variable term
                    term_str = f"{val_str} {feature_clean}"
                
                # Add sign logic for joining later
                rhs_terms.append((is_negative, term_str))
        
        # 4. Assemble the final string
        if not rhs_terms:
            print(f"{lhs} = 0")
            continue
            
        equation_str = f"{lhs} = "
        
        for i, (is_neg, term) in enumerate(rhs_terms):
            if i == 0:
                # First term: sign is attached directly
                if is_neg:
                    equation_str += f"-{term}"
                else:
                    equation_str += f"{term}"
            else:
                # Subsequent terms: sign determines operator
                if is_neg:
                    equation_str += f" - {term}"
                else:
                    equation_str += f" + {term}"
                    
        print(equation_str)
        print("-" * 60)

def parse_term(term_str):
    """
    Robustly cleans sympy-style string representations into readable math.
    Examples:
      '1.0*Derivative(x(t), (t, 2))' -> 'd^2x/dt^2'
      'x(t)**2' -> 'x^2'
      '1.0000000000' -> '1'
    """
    # 1. Remove leading float multipliers usually added by sympy (e.g. "1.0*")
    # We remove it only if it is "1.0*" or "1.0 " at the start. 
    # Real coefficients are handled by the main loop, not here.
    term_str = re.sub(r'^1\.0+\s*\*?\s*', '', term_str)
    
    # 2. Check for pure constant (e.g. "1.00000000000000")
    if re.match(r'^1\.0+$', term_str) or term_str == '1':
        return "1"

    # 3. Replace Derivatives
    # Regex explains: Look for Derivative( VAR(t), ORDER_INFO )
    # ORDER_INFO can be 't' (1st deriv) or '(t, N)' (Nth deriv)
    
    def replace_derivative(match):
        var = match.group(1) # e.g. x
        order_part = match.group(2) # e.g. t or (t, 2)
        
        if order_part == 't':
            return f"d{var}/dt"
        else:
            # Extract number from (t, 2)
            order_match = re.search(r'\d+', order_part)
            order = order_match.group(0) if order_match else '?'
            return f"d^{order}{var}/dt^{order}"

    # Regex pattern for Derivative
    # Matches: Derivative(x(t), t) OR Derivative(x(t), (t, 2))
    # It assumes variables look like x(t) or y(t) inside the derivative.
    pattern = r"Derivative\((\w+)\(t\),\s*((?:\(t,\s*\d+\))|t)\)"
    term_str = re.sub(pattern, replace_derivative, term_str)

    # 4. Clean up remaining functions like x(t) -> x
    term_str = re.sub(r'(\w+)\(t\)', r'\1', term_str)

    # 5. Fix Powers: ** -> ^
    term_str = term_str.replace('**', '^')

    # 6. Fix Multiplication: * -> space (for cleanliness)
    term_str = term_str.replace('*', ' ')
    
    # 7. Final cleanup of whitespace
    term_str = re.sub(r'\s+', ' ', term_str).strip()
    
    return term_str


def visualize_noise_impact(system_class, params, variable='x', noise_levels=[0.05, 0.1], noise_type = 'constant'):
    """
    Visualizes the effect of noise on a single trajectory.
    """
    t_steps = 1000
    dt = 0.01
    
    # 1. Generate Clean "Ground Truth"
    # We use a fixed internal seed in generation to ensure the 'clean' and 'noisy' 
    # trajectories start from the EXACT same initial condition.
    clean_path = generate_data_with_noise(
        data_experiment_name="temp_viz_clean",
        num_experiments_per_param=1,
        num_time_steps=t_steps,
        dt=dt,
        model_class=system_class,
        list_model_params=[params],
        noise_std=0.0,
        noise_type=noise_type,
        seed=42
    )
    
    # Load the clean file we just made
    clean_df = pd.read_csv(f"{clean_path}/solution_params_0_init_cond_0.csv")
    
    plt.figure(figsize=(14, 6))
    plt.plot(clean_df['time'], clean_df[variable], 'k-', linewidth=2, label='Ground Truth (No Noise)')
    
    # 2. Generate and Plot Noisy Versions
    colors = ['tab:blue', 'tab:red', 'tab:orange']
    
    for idx, sigma in enumerate(noise_levels):
        noisy_path = generate_data_with_noise(
            data_experiment_name=f"temp_viz_noise_{sigma}_{noise_type}",
            num_experiments_per_param=1,
            num_time_steps=t_steps,
            dt=dt,
            model_class=system_class,
            list_model_params=[params],
            noise_std=sigma,
            noise_type=noise_type,
            seed=42 # Critical: same seed ensures same underlying trajectory
        )
        
        noisy_df = pd.read_csv(f"{noisy_path}/solution_params_0_init_cond_0.csv")
        
        plt.plot(noisy_df['time'], noisy_df[variable], 
                 linestyle='none', marker='.', markersize=2, alpha=0.6, 
                 color=colors[idx % len(colors)],
                 label=f'Noise $\sigma={sigma}$')

    plt.title(f"Impact of Noise on {system_class.__name__} (Variable: {variable})")
    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- Example Usage: Visualize Lorenz Attractor with Noise ---
# # params for Lorenz: sigma=10, rho=28, beta=8/3
# visualize_noise_impact(
#     system_class=LorenzAttractor, 
#     params={'sigma': 10, 'rho': 28, 'beta': 8.0/3}, 
#     variable='x', 
#     noise_levels=[0.5, 2.0]
# )