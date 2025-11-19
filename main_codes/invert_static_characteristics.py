#!/usr/bin/env python3
import os
import glob
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_coeff_files(folder):
    pattern = os.path.join(folder, 'inv_static_characteristics_*_coeffs.yaml')
    files = sorted(glob.glob(pattern))
    data = []
    for fp in files:
        try:
            with open(fp) as f:
                d = yaml.safe_load(f)
        except Exception as e:
            print(f"Skipping {fp}: failed to load YAML ({e})")
            continue
        if not d:
            print(f"Skipping {fp}: empty YAML")
            continue
        app = d.get('app') or os.path.basename(fp).replace('static_characteristics_','').replace('_coeffs.yaml','')
        coeffs = d.get('coefficients_perf')
        if coeffs is None:
            print(f"Skipping {fp}: no 'coefficients_perf' key")
            continue
        data.append({'path': fp, 'app': app, 'coeffs': coeffs})
    return data


def format_poly_str(coeffs):
    p = np.poly1d(coeffs)
    return str(p)


def main():
    parser = argparse.ArgumentParser(description='Plot progress vs pcap curves from YAML coefficient files')
    parser.add_argument('--folder', default=os.path.join(os.getcwd(), 'main_codes'), help='Folder to search for coeff YAML files')
    parser.add_argument('--xmin', type=float, default=0, help='Minimum progress (x) value')
    parser.add_argument('--xmax', type=float, default=300.0, help='Maximum progress (x) value')
    parser.add_argument('--out', default=os.path.join(os.getcwd(), 'main_codes', 'inv_static_characteristics_all_apps.pdf'), help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show the plot interactively')
    args = parser.parse_args()

    files_data = load_coeff_files(args.folder)
    if not files_data:
        print(f"No coefficient YAML files found in {args.folder} (pattern static_characteristics_*_coeffs.yaml)")
        return

    x = np.linspace(args.xmin, args.xmax, 400)

    plt.figure(figsize=(8, 6))
    for item in files_data:
        coeffs = item['coeffs']
        try:
            poly = np.poly1d(coeffs)
        except Exception as e:
            print(f"Failed to build polynomial for {item['app']}: {e}")
            continue
        y = poly(x)
        plt.plot(x, y, label=item['app'])

    plt.xlabel('Performance [Hz]')
    plt.ylabel('PowerCap [W]')
    plt.title('Performance vs PowerCap fitted polynomials')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

