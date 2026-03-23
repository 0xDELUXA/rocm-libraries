#!/usr/bin/env python3
"""
GEMM Cycle Predictor for gfx950 (CDNA4)

Predicts sclk cycles for a GEMM kernel given problem dimensions and
kernel configuration parsed from Tensile solution names.

Uses peak SCLK for all predictions (no runtime frequency dependency).
Parameters calibrated from microbenchmark measurements.
"""

import csv, re, math

# ============================================================================
# gfx950 Hardware Constants
# ============================================================================
NUM_CUS = 256
NUM_XCDS = 8
CUS_PER_XCD = 32
WAVEFRONT_SIZE = 64

SCLK_GHZ = 2.2
MEM_CLOCK_RATIO = 0.789  # ratio of fabric-to-memory clock domains

# Effective BW from microbenchmark (GB/s)
STREAM_READ_GBS = 6476.0
STREAM_WRITE_GBS = 4759.0
L2_BW_TBS = 17.0

# Derived constants
STREAM_READ_BPC = STREAM_READ_GBS * 1e9 / (SCLK_GHZ * 1e9)
STREAM_WRITE_BPC = STREAM_WRITE_GBS * 1e9 / (SCLK_GHZ * 1e9)
L2_BW_BPC = L2_BW_TBS * 1e12 / (SCLK_GHZ * 1e9)

# MFMA: (mi_m, mi_n, bpe_bits) -> (latency, hw_mi_k, issue_interval)
# Calibrated from microbenchmark (adjusted for loop overhead)
MFMA_TABLE = {
    (16, 16, 16): (36, 32, 14),
    (32, 32, 16): (36, 16, 26),
    (16, 16, 32): (45, 4,  26),
    (32, 32, 32): (45, 2,  26),
}

# ============================================================================
# Heuristic Parameters (calibrated via grid search on benchmark data)
# ============================================================================
H_BW_A = 0.0757;  H_BW_B = -0.00131;  H_BW_C = 0.024
H_BW_CLAMP_LO = 0.02;  H_BW_CLAMP_HI = 0.95

H_VW_BASE = 0.65;  H_VW_SLOPE = 0.05;  H_VW_FLOOR = 0.65;  H_VW_CEIL = 1.05
H_L2_EFFICIENCY = 0.20
H_L2_CAPACITY_PER_XCD = 4 * 1024 * 1024
H_NT_L2_REDUCTION = 1.0
H_DTL_ITER_DISCOUNT = 0.5

H_DISPATCH = 2000
H_PROLOGUE_PER_PGR = 220
H_EPILOGUE_BASE = 300
H_PER_ITER_OVERHEAD = 20
H_STREAMK_REDUCE = 2500

# ============================================================================
# Utility
# ============================================================================
def get_mfma_params(mi_m, mi_n, bpe_bits):
    return MFMA_TABLE.get((mi_m, mi_n, bpe_bits), (36, 32, 14))

def ceil_div(a, b): return (a + b - 1) // b

def dtype_bytes(s):
    if 'bf16' in s or 'f16' in s: return 2
    if 'f32' in s: return 4
    return 2

def bw_scale(n):
    x = max(1, min(n, NUM_CUS))
    return max(H_BW_CLAMP_LO, min(H_BW_CLAMP_HI,
               H_BW_A * math.sqrt(x) + H_BW_B * x + H_BW_C))

def compute_streamk_grid(tiles, k_iters, cu=NUM_CUS):
    if tiles >= cu: return tiles
    for f in [16, 12, 8, 6, 4, 3, 2, 1]:
        if tiles * f <= cu and k_iters // f >= 8: return tiles * f
    return tiles

def parse_solution(sn):
    p = {}
    def ext(pat, d=0):
        m = re.search(pat, sn)
        return int(m.group(1)) if m else d
    mt = re.search(r'MT(\d+)x(\d+)x(\d+)', sn)
    if mt: p['mt_m'],p['mt_n'],p['du'] = int(mt.group(1)),int(mt.group(2)),int(mt.group(3))
    mi = re.search(r'MI(\d+)x(\d+)x(\d+)', sn)
    p['mi_m'] = int(mi.group(1)) if mi else 16
    p['mi_n'] = int(mi.group(2)) if mi else 16
    miwt = re.search(r'MIWT(\d+)_(\d+)', sn)
    p['miwt_m'] = int(miwt.group(1)) if miwt else 1
    p['miwt_n'] = int(miwt.group(2)) if miwt else 1
    wg = re.search(r'WG(\d+)_(\d+)_(\d+)', sn)
    if wg:
        p['wg_m'],p['wg_n'],p['wg_k'] = int(wg.group(1)),int(wg.group(2)),int(wg.group(3))
        p['wave_num'] = max(1, p['wg_m']*p['wg_n']*p['wg_k']//WAVEFRONT_SIZE)
    else: p['wg_m'],p['wg_n'],p['wg_k'],p['wave_num'] = 64,4,1,4
    p['pgr']=ext(r'PGR(\d+)',2); p['gsu']=ext(r'_GSU(\d+)_'); p['sk']=ext(r'SK(\d+)')
    p['grvw_a']=ext(r'GRVWA(\d+)',1); p['grvw_b']=ext(r'GRVWB(\d+)',1)
    p['dtla']=ext(r'DTLA(\d+)'); p['dtlb']=ext(r'DTLB(\d+)')
    p['nta']=ext(r'NTA(\d+)'); p['ntb']=ext(r'NTB(\d+)')
    p['nlca']=ext(r'NLCA(\d+)',1); p['nlcb']=ext(r'NLCB(\d+)',1)
    p['su']=ext(r'_SU(\d+)_')
    return p

# ============================================================================
# L2 Hit Rate
# ============================================================================
def estimate_l2_hit(M, N, K, MT_M, MT_N, DU, bpe, num_active_cus, ntb=0):
    gm = ceil_div(M, MT_M); gn = ceil_div(N, MT_N)
    if gm <= 1 and gn <= 1: return 0.0
    wgm = max(1, int(math.sqrt(CUS_PER_XCD)))
    eff = max(1, ceil_div(num_active_cus, NUM_XCDS))
    sn = min(wgm, gn); sm = max(1, min(ceil_div(eff, sn), gm))
    uA = sm*MT_M*DU*bpe; uB = sn*MT_N*DU*bpe
    tA = uA*sn; tB = uB*sm; tr = max(tA+tB, 1)
    h = max(0.0, (tr-(uA+uB))/tr)
    if uA+uB > H_L2_CAPACITY_PER_XCD: h *= H_L2_CAPACITY_PER_XCD/(uA+uB)
    if ntb > 0: h *= H_NT_L2_REDUCTION
    return min(h, 0.95)

# ============================================================================
# Prediction
# ============================================================================
def predict_cycles(M, N, K, batch, a_type, d_type, sol, freq_mhz=None):
    bpe = dtype_bytes(a_type); bpe_d = dtype_bytes(d_type); bpe_bits = bpe*8
    MT_M,MT_N,DU = sol['mt_m'],sol['mt_n'],sol['du']
    MI_M,MI_N = sol['mi_m'],sol['mi_n']
    MIWT_M,MIWT_N = sol['miwt_m'],sol['miwt_n']
    PGR = sol['pgr']; LSU = sol['wg_k']
    mi_lat, hw_mi_k, mi_iss = get_mfma_params(MI_M, MI_N, bpe_bits)

    gm = ceil_div(M,MT_M); gn = ceil_div(N,MT_N)
    tiles = gm*gn*batch; kpt = ceil_div(K,DU)

    if tiles >= NUM_CUS:
        sf=1; ac=NUM_CUS; ts=ceil_div(tiles,NUM_CUS)
    elif sol['sk']>0 and kpt>=8:
        sg=compute_streamk_grid(tiles,kpt); sf=max(1,ceil_div(sg,tiles))
        nw=tiles*sf; ac=min(nw,NUM_CUS); ts=ceil_div(nw,NUM_CUS)
    else:
        sf=1; ac=min(tiles,NUM_CUS); ts=ceil_div(tiles,NUM_CUS)

    Kps = ceil_div(K,sf); ki = ceil_div(Kps,DU)

    DUw = DU//LSU if LSU>1 else DU
    mk = ceil_div(DUw, hw_mi_k); ns = MIWT_M*MIWT_N; mpw = ns*mk
    rg = ns * mi_iss
    if rg >= mi_lat: Lc = mpw * mi_iss
    else:
        st = mi_lat - rg
        Lc = mk*(ns*mi_iss) + max(0,mk-1)*st

    bpi = (MT_M*DU + MT_N*DU)*bpe
    sc = bw_scale(ac)
    base_bw = STREAM_READ_BPC * sc / ac
    avg_grvw = (sol.get('grvw_a',1)+sol.get('grvw_b',1))/2.0
    vw = max(H_VW_FLOOR, min(H_VW_CEIL, H_VW_BASE+H_VW_SLOPE*avg_grvw))
    l2h = estimate_l2_hit(M,N,K,MT_M,MT_N,DU,bpe,ac,sol.get('ntb',0))
    l2bpc = L2_BW_BPC/ac
    l2b = l2h * max(0, l2bpc-base_bw) * H_L2_EFFICIENCY
    eff_bw = base_bw*vw + l2b
    Lm = bpi/eff_bw if eff_bw>0 else 0

    Li = max(Lc,Lm) if PGR>=1 else Lc+Lm
    dtl = sol.get('dtla',0) or sol.get('dtlb',0)
    pio = H_PER_ITER_OVERHEAD * (H_DTL_ITER_DISCOUNT if dtl else 1.0)
    Lb = ki*(Li+pio)

    Lp = H_PROLOGUE_PER_PGR*PGR
    sa = min(tiles,ac)
    swb = STREAM_WRITE_BPC*bw_scale(sa)/sa
    Ls = (MT_M*MT_N*bpe_d)/swb if swb>0 else 0
    kr = Kps%DU
    Lt = Li*(kr/DU)*1.5 if kr>0 and DU>1 and ki>0 else 0
    Le = Ls+Lt+H_EPILOGUE_BASE
    Llsu = (MT_M*MT_N*4/256+200*LSU) if LSU>1 else 0

    Ltile = Lp+Lb+Le+Llsu
    total = Ltile*ts + H_DISPATCH
    if sf>1:
        rb = MT_M*MT_N*4*sf
        total += rb/256.0*ceil_div(tiles,ac) + H_STREAMK_REDUCE
    return total

# ============================================================================
# Main
# ============================================================================
def main():
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/.cursor/projects/workspace/uploads/origami_latest.csv'
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        freq = float(row['lowest_avg_freq'])
        if freq < 1500: continue
        sol = parse_solution(row['solution_name'])
        if 'mt_m' not in sol: continue
        M,N,K = int(row['m']),int(row['n']),int(row['k'])
        batch = int(row['batch_count'])
        measured = float(row['us']) * SCLK_GHZ * 1000
        predicted = predict_cycles(M,N,K,batch,row['a_type'],row['d_type'],sol)
        ratio = predicted/measured if measured>0 else 0
        results.append({'ratio':ratio,'err':abs(ratio-1)*100})

    results.sort(key=lambda x: x['err'])
    total = len(results)
    med = sorted(r['ratio'] for r in results)[total//2]
    print(f"Total: {total}, Median: {med:.3f}")
    for pct in [5,10,15,20,25,30,40,50,75,100]:
        w = sum(1 for r in results if r['err'] <= pct)
        print(f"  ±{pct:>3}%: {w:>6} ({100*w/total:>5.1f}%)")

if __name__ == '__main__':
    main()
