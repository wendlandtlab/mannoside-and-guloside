import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

# 1. Load and Normalize Data
df = pd.read_csv('free_sugar_timecourse.csv')

# Define the five sugar columns
sugar_cols = ['glucoside', 'alloside', 'galactoside', 'mannoside', 'taloside']

# Normalize mass balance to 100% for each row
# This ensures Sum(G, A, Gal, M, T) = 100 at every time point
row_totals = df[sugar_cols].sum(axis=1)
for col in sugar_cols:
    df[col] = (df[col] / row_totals) * 100

# Save normalized data for your records
df.to_csv('normalized_free_sugar_timecourse.csv', index=False)

t_data = df['time_in_h'].values
y_data_stacked = df[sugar_cols].values

# Custom Colors
colors = {
    'glucoside': '#939393', 'alloside': '#F18F99', 'galactoside': '#FFCC00',
    'mannoside': '#276EBE', 'taloside': '#3B8BD6'
}

# 2. Fully Reversible Kinetic Model
def reaction_model(y, t, k1, k1r, k2, k2r, k3, k3r, k4, k4r, k5, k5r):
    G, A, Gal, M, T = y
    
    # Differential Equations
    dGdt   = -(k1 + k2 + k3) * G + k1r * A + k2r * Gal + k3r * M
    dAdt   = k1 * G - k1r * A
    dGaldt = k2 * G - k2r * Gal - k4 * Gal + k4r * T
    dTdt   = k4 * Gal - k4r * T - k5 * T + k5r * M
    dMdt   = k3 * G - k3r * M + k5 * T - k5r * M
    
    return [dGdt, dAdt, dGaldt, dMdt, dTdt]

# 3. Global Optimizer Setup
def objective(params):
    # Initial conditions are the first row of normalized data
    y0 = y_data_stacked[0]
    try:
        sol = odeint(reaction_model, y0, t_data, args=tuple(params))
        # Return Sum of Squared Errors
        return np.sum((y_data_stacked - sol)**2)
    except:
        return 1e12 # Penalty for unstable integrations

# Bounds for the 10 rate constants (0 to 5.0 h⁻¹)
bounds = [(0, 5.0)] * 10 

print("Data normalized to 100%. Starting Global Optimization...")
result = differential_evolution(objective, bounds, strategy='best1bin', 
                                popsize=12, tol=0.01, mutation=(0.5, 1))

popt = result.x
labels = ['k1 (G->A)', 'k1r (A->G)', 'k2 (G->Gal)', 'k2r (Gal->G)', 
          'k3 (G->M)', 'k3r (M->G)', 'k4 (Gal->T)', 'k4r (T->Gal)',
          'k5 (T->M)', 'k5r (M->T)']

# 4. Results and Error Analysis
print("\n--- Optimized Global Rate Constants ---")
for name, val in zip(labels, popt):
    print(f"{name:15}: {val:.4f}")

# Calculate Final Error
y_fit_at_data = odeint(reaction_model, y_data_stacked[0], t_data, args=tuple(popt))
rmsd = np.sqrt(mean_squared_error(y_data_stacked, y_fit_at_data))
# Since data is normalized to 100, RMSD is now in 'percentage points of total mass'
print(f"\nAverage Fit Error (RMSD): {rmsd:.2f} %-points")

# 5. Visualization
t_fine = np.linspace(0, max(t_data), 200)
sol_fine = odeint(reaction_model, y_data_stacked[0], t_fine, args=tuple(popt))

fig, ax = plt.subplots(figsize=(12, 7))
for i, name in enumerate(sugar_cols):
    ax.plot(t_fine, sol_fine[:, i], color=colors[name], lw=2.5, label=f'{name.capitalize()} Fit')
    ax.scatter(t_data, y_data_stacked[:, i], color=colors[name], s=50, edgecolor='none', alpha=0.7)

ax.set_xlabel('Time (h)', fontsize=12)
ax.set_ylabel('Concentration (% of total sugar)', fontsize=12)
ax.set_title('Global Kinetic Fit: Normalized Mass Balance (100%)', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
