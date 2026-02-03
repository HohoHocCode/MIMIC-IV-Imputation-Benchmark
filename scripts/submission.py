
import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Robust Normalization
# -----------------------------------------------------------------------------

def _bitcast_to_u8(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8) if x.dtype != torch.uint8 else x

def _normalize_sf(sf, mn, k, l):
    if sf is None or sf.numel() == 0:
        return torch.ones((l, mn, k // 16), device='cuda', dtype=torch.float8_e4m3fn)
    
    if sf.dim() == 6:
        sh = sf.shape
        # (32, 4, MN/128, 4, K/?, L) -> (L, MN, K_blk)
        s = sf.permute(5, 2, 4, 0, 1, 3).contiguous()
        s = s.reshape(l, sh[2] * 128, -1)
        if s.shape[1] > mn:
            s = s[:, :mn, :]
        elif s.shape[1] < mn:
            pad = torch.ones((l, mn - s.shape[1], s.shape[2]), device=sf.device, dtype=sf.dtype)
            s = torch.cat([s, pad], dim=1)
    elif sf.dim() == 3:
        sh = sf.shape
        if sh[0] == l and sh[1] == mn:
             s = sf
        elif sh[2] == l and sh[0] == mn:
             s = sf.permute(2, 0, 1)
        else:
             s = sf
    else:
        s = sf
        
    # Granularity 32 -> 16
    if s.shape[2] * 2 == k // 16:
        s = s.repeat_interleave(2, dim=2)
    
    return s.to(torch.float8_e4m3fn).contiguous()

# -----------------------------------------------------------------------------
# Production Blackwell Kernel
# -----------------------------------------------------------------------------

@triton.jit
def fused_dual_gemm_cutlass_kernel(
    a_ptr, b1_ptr, b2_ptr, c_ptr,
    sfa_ptr, sfb1_ptr, sfb2_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_b1l, stride_b1n, stride_b1k,
    stride_b2l, stride_b2n, stride_b2k,
    stride_cl, stride_cm, stride_cn,
    stride_sf_al, stride_sf_am, stride_sf_ak,
    stride_sf_b1l, stride_sf_b1n, stride_sf_b1k,
    stride_sf_b2l, stride_sf_b2n, stride_sf_b2k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 Locality
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    pid_batch = tl.program_id(axis=1)

    # Accumulators (dot_scaled requires FP32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Range Indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = rm[:, None] < M
    mask_n = rn[:, None] < N

    # K-Loop
    for k in range(0, K, BLOCK_K):
        # Local Indices
        rk = tl.arange(0, BLOCK_K // 2)
        rs = tl.arange(0, BLOCK_K // 16)
        
        # Load A (M, K)
        pts_a = a_ptr + pid_batch * stride_al + rm[:, None] * stride_am + (k // 2 + rk[None, :]) * stride_ak
        a = tl.load(pts_a, mask=mask_m, other=0)
        
        # Load B1, B2 (N, K)
        pts_b1 = b1_ptr + pid_batch * stride_b1l + rn[:, None] * stride_b1n + (k // 2 + rk[None, :]) * stride_b1k
        pts_b2 = b2_ptr + pid_batch * stride_b2l + rn[:, None] * stride_b2n + (k // 2 + rk[None, :]) * stride_b2k
        b1 = tl.load(pts_b1, mask=mask_n, other=0)
        b2 = tl.load(pts_b2, mask=mask_n, other=0)
        
        # Load Scales
        sf_k = k // 16
        pts_sfa = sfa_ptr + pid_batch * stride_sf_al + rm[:, None] * stride_sf_am + (sf_k + rs[None, :]) * stride_sf_ak
        pts_sfb1 = sfb1_ptr + pid_batch * stride_sf_b1l + rn[:, None] * stride_sf_b1n + (sf_k + rs[None, :]) * stride_sf_b1k
        pts_sfb2 = sfb2_ptr + pid_batch * stride_sf_b2l + rn[:, None] * stride_sf_b2n + (sf_k + rs[None, :]) * stride_sf_b2k
        
        sa = tl.load(pts_sfa, mask=mask_m, other=1.0)
        sb1 = tl.load(pts_sfb1, mask=mask_n, other=1.0)
        sb2 = tl.load(pts_sfb2, mask=mask_n, other=1.0)
        
        # Blackwell scaled dot
        # Note: we use .T on weights
        acc1 = tl.dot_scaled(a, sa, "e2m1", b1.T, sb1, "e2m1", acc1)
        acc2 = tl.dot_scaled(a, sa, "e2m1", b2.T, sb2, "e2m1", acc2)

    # Epilogue: SiLU
    res = (acc1 * tl.sigmoid(acc1)) * acc2
    
    # Store
    pts_c = c_ptr + pid_batch * stride_cl + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(pts_c, res.to(tl.float16), mask=(mask_m & (rn[None, :] < N)))

# -----------------------------------------------------------------------------
# Host Wrapper
# -----------------------------------------------------------------------------

@torch.no_grad()
def custom_kernel(data):
    a, b1, b2, sfa, sfb1, sfb2, sfa_p, sfb1_p, sfb2_p, c = data
    M, K2, L = a.shape
    K, N = K2 * 2, b1.shape[0]
    
    a_u8, b1_u8, b2_u8 = _bitcast_to_u8(a), _bitcast_to_u8(b1), _bitcast_to_u8(b2)
    
    # Robust normalization
    sfa_v = _normalize_sf(sfa if sfa.numel() > 0 else sfa_p, M, K, L)
    sfb1_v = _normalize_sf(sfb1 if sfb1.numel() > 0 else sfb1_p, N, K, L)
    sfb2_v = _normalize_sf(sfb2 if sfb2.numel() > 0 else sfb2_p, N, K, L)
    
    if c is None:
        c = torch.empty((M, N, L), device=a.device, dtype=torch.float16)

    # Best config: BLOCK 128x64, BLOCK_K=128
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), L)
    
    fused_dual_gemm_cutlass_kernel[grid](
        a_u8, b1_u8, b2_u8, c, sfa_v, sfb1_v, sfb2_v,
        M, N, K,
        a_u8.stride(2), a_u8.stride(0), a_u8.stride(1),
        b1_u8.stride(2), b1_u8.stride(0), b1_u8.stride(1),
        b2_u8.stride(2), b2_u8.stride(0), b2_u8.stride(1),
        c.stride(2), c.stride(0), c.stride(1),
        sfa_v.stride(0), sfa_v.stride(1), sfa_v.stride(2),
        sfb1_v.stride(0), sfb1_v.stride(1), sfb1_v.stride(2),
        sfb2_v.stride(0), sfb2_v.stride(1), sfb2_v.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        num_warps=4, num_stages=3  # Best config
    )
    return c
