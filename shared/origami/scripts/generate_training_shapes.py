#!/usr/bin/env python3
"""
Generate GEMM training shapes for ML kernel recommender.

Produces a CSV file with (M, N, K, batch, category_id) for each sample,
structured with tile-boundary, cache-alignment, and odd-value probes.

Usage:
  # Default: ~1M shapes
  python generate_training_shapes.py -o shapes.csv

  # 2M shapes
  python generate_training_shapes.py -o shapes.csv --total 2000000

  # 500K shapes, cap M/N at 65536
  python generate_training_shapes.py -o shapes.csv --total 500000 --cap-mn 65536

  # Specific categories only
  python generate_training_shapes.py -o shapes.csv --categories 0,12,62,124

  # With ML features included
  python generate_training_shapes.py -o shapes.csv --total 2000000 --features
"""

import argparse
import csv
import math
import sys
from typing import List, Tuple

# ── Range boundaries (must match categorization.hpp) ──────────────────────────

MN_BOUNDS = [64, 256, 1024, 4096]  # upper bounds for tiny..large; xlarge is inf
K_BOUNDS  = [128, 512, 2048, 8192]

NUM_MN = len(MN_BOUNDS) + 1  # 5
NUM_K  = len(K_BOUNDS) + 1   # 5
NUM_CATEGORIES = NUM_MN * NUM_MN * NUM_K  # 125

# ── Tile sizes from origami heuristics (CMS kernel configs) ───────────────────

MN_TILES = [32, 64, 96, 128, 160, 192, 208, 224, 256]
K_TILES  = [16, 32, 64, 128, 256, 512]

# Primes for odd/non-power-of-2 probes
PRIMES = [3, 7, 13, 17, 31, 37, 41, 47, 53, 61, 67, 71, 73, 79, 89, 97]

# Cache-line aligned values (128B / 2B_BF16 = 64 elements)
CACHE_ALIGNED = [64, 128, 192, 320, 384, 448, 576, 640, 768, 896]


def mn_range(idx: int, cap: int) -> Tuple[int, int]:
    lo = 1 if idx == 0 else MN_BOUNDS[idx - 1] + 1
    hi = MN_BOUNDS[idx] if idx < len(MN_BOUNDS) else cap
    return lo, hi


def k_range(idx: int, cap: int) -> Tuple[int, int]:
    lo = 1 if idx == 0 else K_BOUNDS[idx - 1] + 1
    hi = K_BOUNDS[idx] if idx < len(K_BOUNDS) else cap
    return lo, hi


def cat_id(mi: int, ni: int, ki: int) -> int:
    return mi * NUM_MN * NUM_K + ni * NUM_K + ki


def generate_dim_points(lo: int, hi: int, log_count: int,
                        tile_sizes: List[int]) -> List[int]:
    """Generate structured sample points for one dimension."""
    points = set()
    lo = max(lo, 1)
    hi = max(hi, lo)

    log_lo = math.log2(lo)
    log_hi = math.log2(hi)

    # Layer 1: log-uniform base grid
    for i in range(log_count):
        t = 0.5 if log_count == 1 else i / (log_count - 1)
        val = round(2 ** (log_lo + t * (log_hi - log_lo)))
        val = max(lo, min(hi, val))
        points.add(val)

    # Layer 2: tile-boundary neighbors around each base point
    base_points = list(points)
    for mt in tile_sizes:
        for base in base_points:
            below = (base // mt) * mt
            above = below + mt
            for boundary in [below, above]:
                for offset in [-1, 0, 1]:
                    v = boundary + offset
                    if lo <= v <= hi:
                        points.add(v)
        # Also small multiples
        for k in range(1, 5):
            boundary = mt * k
            for offset in [-1, 0, 1]:
                v = boundary + offset
                if lo <= v <= hi:
                    points.add(v)

    # Layer 3: odd / prime values
    for p in PRIMES:
        if lo <= p <= hi:
            points.add(p)
        for mul in [10, 100, 1000, 10000]:
            v = p * mul
            if lo <= v <= hi:
                points.add(v)

    # Layer 4: cache-alignment probes
    for ca in CACHE_ALIGNED:
        if lo <= ca <= hi:
            points.add(ca)
        if lo <= ca + 1 <= hi:
            points.add(ca + 1)

    return sorted(points)


def subsample(pts: List[int], target: int) -> List[int]:
    """Subsample a sorted list to target size, preserving endpoints."""
    if len(pts) <= target:
        return pts
    result = [pts[0]]
    for i in range(1, target - 1):
        idx = i * (len(pts) - 1) // (target - 1)
        result.append(pts[idx])
    result.append(pts[-1])
    return sorted(set(result))


def compute_ml_features(m: int, n: int, k: int, bpe: float = 2.0) -> dict:
    dm, dn, dk = max(m, 1.0), max(n, 1.0), max(k, 1.0)
    denom = (dm * dk + dk * dn + dm * dn) * bpe
    ai = 2.0 * dm * dn * dk / denom if denom > 0 else 0.0
    return {
        "log2_m": math.log2(dm),
        "log2_n": math.log2(dn),
        "log2_k": math.log2(dk),
        "arithmetic_intensity": ai,
        "mn_aspect_ratio": math.log2(dm / dn),
        "k_mn_ratio": math.log2(dk / math.sqrt(dm * dn)),
        "log2_mn_tiles": math.log2(dm * dn / (256.0 * 256.0)),
    }


def generate_category_samples(mi: int, ni: int, ki: int,
                              max_per_cat: int, log_count: int,
                              cap_mn: int, cap_k: int) -> List[Tuple[int, int, int]]:
    m_lo, m_hi = mn_range(mi, cap_mn)
    n_lo, n_hi = mn_range(ni, cap_mn)
    k_lo, k_hi = k_range(ki, cap_k)

    m_pts = generate_dim_points(m_lo, m_hi, log_count, MN_TILES)
    n_pts = generate_dim_points(n_lo, n_hi, log_count, MN_TILES)
    k_pts = generate_dim_points(k_lo, k_hi, log_count, K_TILES)

    per_dim = max(int(round(max_per_cat ** (1.0 / 3.0))), 4)
    m_pts = subsample(m_pts, per_dim)
    n_pts = subsample(n_pts, per_dim)
    k_pts = subsample(k_pts, per_dim)

    samples = []
    for m in m_pts:
        for n in n_pts:
            for k in k_pts:
                samples.append((m, n, k))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate GEMM training shapes for ML kernel recommender")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("--total", type=int, default=1000000,
                        help="Target total number of shapes (default: 1M)")
    parser.add_argument("--log-points", type=int, default=8,
                        help="Log-spaced base points per dimension (default: 8)")
    parser.add_argument("--cap-mn", type=int, default=131072,
                        help="Upper cap for M/N xlarge range (default: 131072)")
    parser.add_argument("--cap-k", type=int, default=32768,
                        help="Upper cap for K xlarge range (default: 32768)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size to include in output (default: 1)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category IDs to generate (default: all)")
    parser.add_argument("--features", action="store_true",
                        help="Include ML features in output columns")
    args = parser.parse_args()

    if args.categories:
        cat_ids = [int(x) for x in args.categories.split(",")]
    else:
        cat_ids = list(range(NUM_CATEGORIES))

    max_per_cat = max(args.total // len(cat_ids), 64)

    total = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["M", "N", "K", "batch", "category_id"]
        if args.features:
            header += ["log2_m", "log2_n", "log2_k",
                       "arithmetic_intensity", "mn_aspect_ratio",
                       "k_mn_ratio", "log2_mn_tiles"]
        writer.writerow(header)

        for cid in cat_ids:
            ki = cid % NUM_K
            ni = (cid // NUM_K) % NUM_MN
            mi = cid // (NUM_K * NUM_MN)

            samples = generate_category_samples(
                mi, ni, ki, max_per_cat, args.log_points,
                args.cap_mn, args.cap_k)

            for m, n, k in samples:
                row = [m, n, k, args.batch, cid]
                if args.features:
                    feat = compute_ml_features(m, n, k)
                    row += [f"{feat['log2_m']:.4f}",
                            f"{feat['log2_n']:.4f}",
                            f"{feat['log2_k']:.4f}",
                            f"{feat['arithmetic_intensity']:.4f}",
                            f"{feat['mn_aspect_ratio']:.4f}",
                            f"{feat['k_mn_ratio']:.4f}",
                            f"{feat['log2_mn_tiles']:.4f}"]
                writer.writerow(row)
                total += 1

    print(f"Generated {total:,} shapes across {len(cat_ids)} categories")
    print(f"Output: {args.output}")
    print(f"  Per category: ~{total // len(cat_ids):,}")


if __name__ == "__main__":
    main()
