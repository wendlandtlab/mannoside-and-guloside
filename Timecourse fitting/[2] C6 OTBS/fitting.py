import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ==========================================
# 1. Define the Kinetic Model
# ==========================================
def reaction_model(t, y, k1, k_inv1, k2, k_inv2, k3, k_inv3, k4, k_inv4, k5, k_inv5):
    """
    Differential equations for the 5-species reaction network.
    Now including reverse rates from Taloside.
    y = [glucoside, mannoside, alloside, taloside, galactoside]
    """
    G, M, A, T, Gal = y
    
    # Differential Equations
    dG_dt = -k1*G - k2*G - k3*G + k_inv1*A + k_inv2*M + k_inv3*Gal
    dM_dt = k2*G - k_inv2*M - k4*M + k_inv4*T
    dA_dt = k1*G - k_inv1*A
    dT_dt = k4*M - k_inv4*T + k5*Gal - k_inv5*T
    dGal_dt = k3*G - k_inv3*Gal - k5*Gal + k_inv5*T
    
    return [dG_dt, dM_dt, dA_dt, dT_dt, dGal_dt]

# ==========================================
# 2. Define the Objective Function for Fitting
# ==========================================
def calculate_residuals(params, df, runs, species):
    """
    Simulates the model for all runs given a set of parameters and 
    returns the flattened residuals for global optimization.
    """
    k1, k_inv1, k2, k_inv2, k3, k_inv3, k4, k_inv4, k5, k_inv5 = params
    all_residuals = []
    
    for run in runs:
        # Extract data for this specific run
        run_data = df[df['Run'] == run].sort_values('time_in_h')
        t_eval = run_data['time_in_h'].values
        y_obs = run_data[species].values
        
        # Initial conditions (from the first timepoint)
        y0 = y_obs[0]
        
        # Integrate ODEs
        sol = solve_ivp(reaction_model, 
                        (t_eval[0], t_eval[-1]), 
                        y0, 
                        t_eval=t_eval, 
                        args=(k1, k_inv1, k2, k_inv2, k3, k_inv3, k4, k_inv4, k5, k_inv5), 
                        method='LSODA')
        
        if sol.success:
            y_pred = sol.y.T # Transpose to match (N_timepoints, 5_species)
            residuals = y_obs - y_pred
            all_residuals.append(residuals.flatten())
        else:
            # If integration fails, return a large penalty
            all_residuals.append(np.ones_like(y_obs).flatten() * 1e6)
            
    return np.concatenate(all_residuals)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    file_name = "mono_TBS_timecourse.xlsx"
    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Ensure the file is in the same directory.")
        return

    # Strip hidden spaces from the column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Handle capitalization differences for 'Run'
    if 'Run' not in df.columns:
        if 'run' in df.columns:
            df.rename(columns={'run': 'Run'}, inplace=True)
        else:
            print("Columns found:", df.columns.tolist())
            print("\nCRITICAL ERROR: Could not find the 'Run' column.")
            return

    runs = df['Run'].unique()
    species = ['glucoside', 'mannoside', 'alloside', 'taloside', 'galactoside']
    
    # Initial guesses for 10 rate constants: 
    # [k1, k_inv1, k2, k_inv2, k3, k_inv3, k4, k_inv4, k5, k_inv5]
    initial_guess = [0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.01, 0.05, 0.01]
    
    # Bounds to ensure rate constants are strictly positive
    bounds = (0, np.inf)

    print("Running global fit across all datasets... This may take a moment.")
    
    # Perform Global Fitting via Least Squares
    result = least_squares(calculate_residuals, 
                           initial_guess, 
                           args=(df, runs, species), 
                           bounds=bounds,
                           method='trf')
    
    # Extract optimized parameters
    opt_k1, opt_k_inv1, opt_k2, opt_k_inv2, opt_k3, opt_k_inv3, opt_k4, opt_k_inv4, opt_k5, opt_k_inv5 = result.x
    
    # Calculate Global RMSD
    global_residuals = result.fun
    rmsd = np.sqrt(np.mean(global_residuals**2))
    
    print("\n--- Optimized Pseudo-1st-Order Rate Constants ---")
    print(f"k1 (G -> A):      {opt_k1:.4f} h^-1")
    print(f"k_inv1 (A -> G):  {opt_k_inv1:.4f} h^-1")
    print(f"k2 (G -> M):      {opt_k2:.4f} h^-1")
    print(f"k_inv2 (M -> G):  {opt_k_inv2:.4f} h^-1")
    print(f"k3 (G -> Gal):    {opt_k3:.4f} h^-1")
    print(f"k_inv3 (Gal -> G):{opt_k_inv3:.4f} h^-1")
    print(f"k4 (M -> T):      {opt_k4:.4f} h^-1")
    print(f"k_inv4 (T -> M):  {opt_k_inv4:.4f} h^-1")
    print(f"k5 (Gal -> T):    {opt_k5:.4f} h^-1")
    print(f"k_inv5 (T -> Gal):{opt_k_inv5:.4f} h^-1")
    print(f"Global RMSD:      {rmsd:.4f}")

    # ==========================================
    # 4. Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(1, len(runs), figsize=(20, 6), sharey=True)
    if len(runs) == 1:
        axes = [axes]
        
    # Styling for the 5 species
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    linestyles = ['-', '--', '-.', ':', '-']

    for idx, run in enumerate(runs):
        ax = axes[idx]
        run_data = df[df['Run'] == run].sort_values('time_in_h')
        
        t_obs = run_data['time_in_h'].values
        y_obs = run_data[species].values
        y0 = y_obs[0]
        
        # Generate smooth time grid for drawing the fitted lines
        t_smooth = np.linspace(t_obs[0], t_obs[-1], 200)
        sol = solve_ivp(reaction_model, 
                        (t_smooth[0], t_smooth[-1]), 
                        y0, 
                        t_eval=t_smooth, 
                        args=(opt_k1, opt_k_inv1, opt_k2, opt_k_inv2, opt_k3, opt_k_inv3, opt_k4, opt_k_inv4, opt_k5, opt_k_inv5), 
                        method='LSODA')
        
        for s_idx, sp in enumerate(species):
            # Plot Scatter Data
            ax.scatter(t_obs, y_obs[:, s_idx], color=colors[s_idx], 
                       label=f'{sp.capitalize()} (obs)' if idx == 0 else "", 
                       marker=markers[s_idx])
            # Plot Fitted Lines
            ax.plot(t_smooth, sol.y[s_idx], color=colors[s_idx], 
                    linestyle=linestyles[s_idx], 
                    label=f'{sp.capitalize()} (fit)' if idx == 0 else "")
        
        ax.set_title(f'Run {run}')
        ax.set_xlabel('Time (h)')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Concentration / Molar Fraction')
            ax.legend(fontsize=8, loc='best', ncol=2)

    # Add text box for 10 parameters + RMSD
    param_text = (
        f"Global Fit Results:\n"
        f"k1 (G->A) = {opt_k1:.4f}\n"
        f"k_inv1 (A->G) = {opt_k_inv1:.4f}\n"
        f"k2 (G->M) = {opt_k2:.4f}\n"
        f"k_inv2 (M->G) = {opt_k_inv2:.4f}\n"
        f"k3 (G->Gal) = {opt_k3:.4f}\n"
        f"k_inv3 (Gal->G) = {opt_k_inv3:.4f}\n"
        f"k4 (M->T) = {opt_k4:.4f}\n"
        f"k_inv4 (T->M) = {opt_k_inv4:.4f}\n"
        f"k5 (Gal->T) = {opt_k5:.4f}\n"
        f"k_inv5 (T->Gal) = {opt_k_inv5:.4f}\n\n"
        f"RMSD = {rmsd:.4f}"
    )
    plt.figtext(0.99, 0.5, param_text, fontsize=10, va='center', ha='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle("Global Fitting: Reversible Taloside Model", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85) # Leave space for the text box
    
    # Save the figure to your cluster
    plt.savefig("fitting_results_reversible.png", dpi=300)
    print("Plot successfully saved as 'fitting_results_reversible.png'.")

if __name__ == "__main__":
    main()
