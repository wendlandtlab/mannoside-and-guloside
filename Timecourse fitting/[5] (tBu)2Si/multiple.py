import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution

# ==========================================
# 1. Define the Kinetic Model (Catalyst Added)
# ==========================================
def reaction_model(t, y, k1, k_inv1, k2, k_inv2, Cat):
    """
    Differential equations for the reaction network including catalyst.
    y = [glucoside, alloside, mannoside]
    Cat = Constant catalyst concentration for the run
    """
    G, A, M = y
    dG_dt = -k1 * G * Cat - k2 * G * Cat + k_inv1 * A * Cat + k_inv2 * M * Cat
    dA_dt = k1 * G * Cat - k_inv1 * A * Cat
    dM_dt = k2 * G * Cat - k_inv2 * M * Cat
    return [dG_dt, dA_dt, dM_dt]

# ==========================================
# 2. Define the Objective Functions
# ==========================================
def calculate_residuals(params, df, runs):
    k1, k_inv1, k2, k_inv2 = params
    all_residuals = []

    for run in runs:
        run_data = df[df['Run'] == run].sort_values('time_in_h')
        t_eval = run_data['time_in_h'].values
        y_obs = run_data[['glucoside', 'alloside', 'mannoside']].values
        y0 = y_obs[0]
        
        # Extract catalyst concentration for this specific run
        Cat = run_data['catalyst'].iloc[0]

        sol = solve_ivp(reaction_model,
                        (t_eval[0], t_eval[-1]),
                        y0,
                        t_eval=t_eval,
                        args=(k1, k_inv1, k2, k_inv2, Cat), 
                        method='LSODA')

        if sol.success:
            y_pred = sol.y.T 
            residuals = y_obs - y_pred
            all_residuals.append(residuals.flatten())
        else:
            all_residuals.append(np.ones_like(y_obs).flatten() * 1e6)

    return np.concatenate(all_residuals)

def calculate_ssr(params, df, runs):
    residuals = calculate_residuals(params, df, runs)
    return np.sum(residuals**2)

# ==========================================
# 3. Main Execution & Fitting
# ==========================================
def main():
    file_name = "protected_glucose_timecourse.xlsx"
    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return

    # Check for catalyst column
    if 'catalyst' not in df.columns:
        raise ValueError("Error: Your dataset must contain a 'catalyst' column with the concentration for each run.")

    runs = df['Run'].unique()

    # Differential Evolution bounds (adjust max if true 2nd-order rates are > 10)
    bounds = [(0, 10), (0, 10), (0, 10), (0, 10)]

    print("Step 1: Running Differential Evolution to find the global minimum...")
    print("(Using 'rand1bin' strategy for maximum global exploration. This will take longer to compute...)")
    
    de_result = differential_evolution(calculate_ssr, 
                                       bounds, 
                                       args=(df, runs), 
                                       strategy='rand1bin',   # Maximize global search
                                       popsize=30,            # Larger population to avoid missing valleys
                                       mutation=(0.5, 1.5),   # Wider mutation for better landscape jumping
                                       recombination=0.7)
    
    print("\nStep 2: Polishing the result and calculating error matrices...")
    ls_bounds = (0, np.inf)
    result = least_squares(calculate_residuals,
                           de_result.x,
                           args=(df, runs),
                           bounds=ls_bounds,
                           method='trf')

    opt_k1, opt_k_inv1, opt_k2, opt_k_inv2 = result.x
    global_residuals = result.fun

    # Error Analysis
    rmsd = np.sqrt(np.mean(global_residuals**2))
    dof = len(global_residuals) - len(result.x)
    sigma_sq = np.sum(global_residuals**2) / dof
    
    try:
        J = result.jac
        cov_matrix = np.linalg.inv(J.T @ J) * sigma_sq
        std_errors = np.sqrt(np.diagonal(cov_matrix))
    except np.linalg.LinAlgError:
        print("Warning: Could not invert Jacobian to calculate standard errors.")
        std_errors = [np.nan] * 4

    print("\n--- Optimized 2nd-Order Rate Constants (per M per h) ---")
    print(f"Glucoside -> Alloside (k1):       {opt_k1:.4f} ± {std_errors[0]:.4f}")
    print(f"Alloside -> Glucoside (k_inv1):   {opt_k_inv1:.4f} ± {std_errors[1]:.4f}")
    print(f"Glucoside -> Mannoside (k2):      {opt_k2:.4f} ± {std_errors[2]:.4f}")
    print(f"Mannoside -> Glucoside (k_inv2):  {opt_k_inv2:.4f} ± {std_errors[3]:.4f}")
    print(f"Global RMSD:                      {rmsd:.4f}")

    # Export Metrics to Excel
    metrics_data = {
        'Parameter': ['k1 (G->A)', 'k_inv1 (A->G)', 'k2 (G->M)', 'k_inv2 (M->G)'],
        'Optimized_Value': [opt_k1, opt_k_inv1, opt_k2, opt_k_inv2],
        'Standard_Error': std_errors,
        'Global_RMSD': [rmsd, '', '', '']
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel('error_analysis_metrics.xlsx', index=False)

    # ==========================================
    # 4. Plotting and Saving Individual Results
    # ==========================================
    colors = {'glucoside': '#1F77B4', 'alloside': '#FF7F0E', 'mannoside': '#2CA02C'}

    param_text = (
        f"Global 2nd-Order Rates:\n"
        f"k1 (G->A) = {opt_k1:.4f} ± {std_errors[0]:.4f}\n"
        f"k_inv1 (A->G) = {opt_k_inv1:.4f} ± {std_errors[1]:.4f}\n"
        f"k2 (G->M) = {opt_k2:.4f} ± {std_errors[2]:.4f}\n"
        f"k_inv2 (M->G) = {opt_k_inv2:.4f} ± {std_errors[3]:.4f}\n"
        f"Global RMSD = {rmsd:.4f}"
    )

    for idx, run in enumerate(runs):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        run_data = df[df['Run'] == run].sort_values('time_in_h')
        t_obs = run_data['time_in_h'].values
        y_obs = run_data[['glucoside', 'alloside', 'mannoside']].values
        y0 = y_obs[0]
        Cat = run_data['catalyst'].iloc[0]

        t_smooth = np.linspace(t_obs[0], t_obs[-1], 200)
        sol = solve_ivp(reaction_model,
                        (t_smooth[0], t_smooth[-1]),
                        y0,
                        t_eval=t_smooth,
                        args=(opt_k1, opt_k_inv1, opt_k2, opt_k_inv2, Cat),
                        method='LSODA')

        ax.scatter(t_obs, y_obs[:, 0], color=colors['glucoside'], label='Glucoside (obs)', marker='o')
        ax.scatter(t_obs, y_obs[:, 1], color=colors['alloside'], label='Alloside (obs)', marker='s')
        ax.scatter(t_obs, y_obs[:, 2], color=colors['mannoside'], label='Mannoside (obs)', marker='^')

        ax.plot(t_smooth, sol.y[0], color=colors['glucoside'], linestyle='-', label='Glucoside (fit)')
        ax.plot(t_smooth, sol.y[1], color=colors['alloside'], linestyle='--', label='Alloside (fit)')
        ax.plot(t_smooth, sol.y[2], color=colors['mannoside'], linestyle='-.', label='Mannoside (fit)')

        ax.set_title(f'Run {run} - Global Fit Results (Catalyst = {Cat})')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration / Molar Fraction') # Reverted label
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

        plt.figtext(0.95, 0.5, param_text, fontsize=11, va='center', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        
        image_output_name = f'fit_result_run_{run}.jpg'
        plt.savefig(image_output_name, format='jpg', dpi=300, bbox_inches='tight')
        plt.close(fig) 
        
    print("\nAll processing and saving is complete!")

if __name__ == "__main__":
    main()