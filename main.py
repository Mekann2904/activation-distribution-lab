import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
rng = np.random.default_rng(123)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def density_line(samples: np.ndarray, x_min: float, x_max: float, bins: int = 360):
    """Histogram-based density line (centers, density)"""
    hist, edges = np.histogram(samples, bins=bins, range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

# -----------------------------
# Cases (3 patterns of pre-activation X)
# -----------------------------
cases = [
    ("N(0, 0.3^2)  (mostly linear)", 0.0, 0.3),
    ("N(0, 3^2)    (large variance)", 0.0, 3.0),
    ("N(2, 1^2)    (mean shift)", 2.0, 1.0),
]

# Sample once per case so Fig.2 and Fig.3 are consistent
N = 500_000
samples = {}
for label, mu, sigma in cases:
    samples[label] = rng.normal(mu, sigma, size=N)

# -----------------------------
# Fig.1: Pre-activation distributions p_X(x)
# -----------------------------
xgrid = np.linspace(-8, 8, 2000)
plt.figure()
for label, mu, sigma in cases:
    plt.plot(xgrid, normal_pdf(xgrid, mu, sigma), label=label)
plt.axvline(0.0, linestyle=":", label="x=0")
plt.axvline(2.0, linestyle="--", label="x=2 (shift reference)")
plt.title("Fig.1  Pre-activation distributions p_X(x)")
plt.xlabel("x")
plt.ylabel("density")
plt.legend()
plt.savefig("fig1_pre_activation_pX.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# Fig.2: Output distributions for y = tanh(X)
# -----------------------------
plt.figure()
for label, X in samples.items():
    Y = np.tanh(X)
    u, d = density_line(Y, -1.0, 1.0, bins=360)
    sat = np.mean(np.abs(Y) > 0.95) * 100.0
    plt.plot(u, d, label=f"{label} | mean={Y.mean():.3f}, sat(|y|>0.95)={sat:.1f}%")
plt.axvline(0.0, linestyle=":", label="center y=0")
plt.axvline(-0.95, linestyle="--")
plt.axvline(0.95, linestyle="--", label="saturation threshold")
plt.title("Fig.2  Output distributions for y = tanh(X)")
plt.xlabel("y")
plt.ylabel("density")
plt.xlim(-1, 1)
plt.legend()
plt.savefig("fig2_tanh_output_pY.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# Fig.3: Output distributions for y = sigmoid(X)
# -----------------------------
plt.figure()
for label, X in samples.items():
    Y = sigmoid(X)
    u, d = density_line(Y, 0.0, 1.0, bins=360)
    sat = (np.mean(Y < 0.05) + np.mean(Y > 0.95)) * 100.0
    plt.plot(u, d, label=f"{label} | mean={Y.mean():.3f}, sat(y<0.05 or >0.95)={sat:.1f}%")
plt.axvline(0.5, linestyle=":", label="center y=0.5")
plt.axvline(0.05, linestyle="--")
plt.axvline(0.95, linestyle="--", label="saturation thresholds")
plt.title("Fig.3  Output distributions for y = sigmoid(X)")
plt.xlabel("y")
plt.ylabel("density")
plt.xlim(0, 1)
plt.legend()
plt.savefig("fig3_sigmoid_output_pY.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# Fig.4: Derivatives shrink in saturation regions (normalized)
# -----------------------------
x = np.linspace(-8, 8, 3000)
tanh_prime = 1.0 - np.tanh(x) ** 2
sig_prime = sigmoid(x) * (1.0 - sigmoid(x))

tanh_prime_n = tanh_prime / tanh_prime.max()
sig_prime_n = sig_prime / sig_prime.max()

plt.figure()
plt.plot(x, tanh_prime_n, label="tanh'(x) / max")
plt.plot(x, sig_prime_n, label="sigmoid'(x) / max")
plt.axvline(0.0, linestyle=":", label="x=0")
plt.title("Fig.4  Derivatives shrink in saturation regions")
plt.xlabel("x")
plt.ylabel("normalized derivative")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.savefig("fig4_derivative_shrink.png", dpi=200, bbox_inches="tight")
plt.show()

print("Saved:")
print("  fig1_pre_activation_pX.png")
print("  fig2_tanh_output_pY.png")
print("  fig3_sigmoid_output_pY.png")
print("  fig4_derivative_shrink.png")

