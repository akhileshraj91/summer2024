import yaml, os
def format_poly(coeffs, var='x', precision=6):
    degree = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        p = degree - i
        if abs(c) < 10**(-precision): 
            continue
        sign = '-' if c < 0 else ('+' if parts else '')
        coeff_str = f"{abs(c):.{precision}g}"
        if p == 0:
            parts.append(f"{sign} {coeff_str}")
        elif p == 1:
            parts.append(f"{sign} {coeff_str}*{var}")
        else:
            parts.append(f"{sign} {coeff_str}*{var}^{p}")
    return ' '.join(parts).lstrip('+ ').replace('+ -','- ')

# Example:
coeffs = [0.012, -1.23, 45.6]
print(format_poly(coeffs))   # -> "0.012*x^2 - 1.23*x + 45.6"
path = os.path.join('main_codes', 'static_characteristics_ones-npb-cg_coeffs.yaml')
with open(path) as f:
    data = yaml.safe_load(f)
coeffs_perf = data['coefficients_perf']
coeffs_pow  = data['coefficients_pow']
print("Performance equation:", format_poly(coeffs_perf))
print("Power equation:      ", format_poly(coeffs_pow))