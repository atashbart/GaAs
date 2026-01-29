import numpy as np
import matplotlib.pyplot as plt

# Solar cell parameters from your data
Voc = 1.008312  # Open-circuit voltage (V)
Jsc = 30.52638788  # Short-circuit current density (mA/cm²)
FF = 0.883105  # Fill Factor
eta = 0.271821  # Efficiency
V_MPP = 0.915723  # Voltage at maximum power point (V)
J_MPP = 29.68373326  # Current density at MPP (mA/cm²)

# Diode model parameters (approximated from the data)
Vt = 0.02585  # Thermal voltage at 300K (V)
n = 1.2  # Ideality factor (typical for GaAs)
J0 = 1e-12  # Reverse saturation current (mA/cm²)
Rs = 0.001  # Series resistance (Ω·cm²)
Rsh = 10000  # Shunt resistance (Ω·cm²)

# Single-diode model equation
def solar_cell_current(V, Jsc=Jsc, J0=J0, n=n, Rs=Rs, Rsh=Rsh):
    """
    Single-diode model for solar cell J-V characteristic
    J = Jsc - J0*(exp((V + J*Rs)/(n*Vt)) - 1) - (V + J*Rs)/Rsh
    """
    # Numerical solution using Newton-Raphson method
    J_guess = Jsc  # Initial guess
    
    for _ in range(100):  # Max iterations
        # Diode current
        f = J_guess - Jsc + J0*(np.exp((V + J_guess*Rs)/(n*Vt)) - 1) + (V + J_guess*Rs)/Rsh
        # Derivative
        df = 1 + (J0*Rs/(n*Vt))*np.exp((V + J_guess*Rs)/(n*Vt)) + Rs/Rsh
        
        J_new = J_guess - f/df
        
        if abs(J_new - J_guess) < 1e-10:
            break
        J_guess = J_new
    
    return J_new

# Generate voltage range
V_range = np.linspace(-0.1, Voc + 0.05, 500)
J_range = np.array([solar_cell_current(V) for V in V_range])
P_range = V_range * J_range  # Power density

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot J-V curve
ax1.plot(V_range, J_range, 'b-', linewidth=2.5, label='J-V Curve')
ax1.set_xlabel('Voltage (V)', fontsize=14)
ax1.set_ylabel('Current Density (mA/cm²)', fontsize=14, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

# Mark key points
ax1.plot(Voc, 0, 'ro', markersize=10, label=f'Voc = {Voc:.3f} V')
ax1.plot(0, Jsc, 'go', markersize=10, label=f'Jsc = {Jsc:.2f} mA/cm²')
ax1.plot(V_MPP, J_MPP, 'mo', markersize=10, 
         label=f'MPP: ({V_MPP:.3f} V, {J_MPP:.2f} mA/cm²)')

# Highlight maximum power rectangle
ax1.fill_between([0, V_MPP], 0, J_MPP, alpha=0.2, color='orange',
                 label=f'Max Power: {V_MPP*J_MPP:.2f} mW/cm²')

# Add reference lines
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax1.axvline(x=Voc, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=Jsc, color='g', linestyle='--', alpha=0.5, linewidth=1)

# Create second y-axis for power
ax2 = ax1.twinx()
ax2.plot(V_range, P_range, 'r--', linewidth=1.5, alpha=0.7, label='Power')
ax2.set_ylabel('Power Density (mW/cm²)', fontsize=14, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([0, max(P_range)*1.1])

# Title
ax1.set_title('Current-Voltage (J-V) Characteristics of GaAs p-i-n Solar Cell\n' +
              f'Efficiency (η) = {eta*100:.2f}% | Fill Factor (FF) = {FF*100:.2f}%', 
              fontsize=14, fontweight='bold')

# Set limits
ax1.set_xlim([-0.1, Voc + 0.05])
ax1.set_ylim([-2, Jsc * 1.1])

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# Add text box with cell parameters
textstr = '\n'.join((
    f'Voc = {Voc:.3f} V',
    f'Jsc = {Jsc:.2f} mA/cm²',
    f'FF = {FF*100:.2f}%',
    f'η = {eta*100:.2f}%',
    f'V_MPP = {V_MPP:.3f} V',
    f'J_MPP = {J_MPP:.2f} mA/cm²',
    f'GaAs p-i-n Solar Cell'
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Additional analysis
print("=== SOLAR CELL PERFORMANCE ANALYSIS ===")
print(f"1. Open-Circuit Voltage (Voc): {Voc:.3f} V")
print(f"2. Short-Circuit Current Density (Jsc): {Jsc:.2f} mA/cm²")
print(f"3. Fill Factor (FF): {FF*100:.2f}%")
print(f"4. Efficiency (η): {eta*100:.2f}%")
print(f"5. Maximum Power Point:")
print(f"   - Voltage (V_MPP): {V_MPP:.3f} V ({V_MPP/Voc*100:.1f}% of Voc)")
print(f"   - Current (J_MPP): {J_MPP:.2f} mA/cm² ({J_MPP/Jsc*100:.1f}% of Jsc)")
print(f"   - Power Density: {V_MPP*J_MPP:.3f} mW/cm²")
print(f"6. Theoretical Maximum Power: {Voc*Jsc*FF:.3f} mW/cm²")
print(f"7. GaAs p-i-n Solar Cell with ~2 µm active layer")
print("\n=== PERFORMANCE EVALUATION ===")
print(f"✓ Excellent efficiency (>27%)")
print(f"✓ Outstanding fill factor (>88%)")
print(f"✓ Very high current density (>30 mA/cm²)")
print(f"✓ Competitive with state-of-the-art GaAs cells")
