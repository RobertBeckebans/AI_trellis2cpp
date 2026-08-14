"""Shared plumbing for the PyTorch reference dumps.

The reference environment is deliberately stock PyTorch (docker/Dockerfile.ref):
none of the custom CUDA extensions (flash-attn, FlexGEMM, cumesh, o-voxel) are
installed. This module makes the trellis2 package importable and usable in that
environment:

  * stubs out `cumesh` (only needed by Mesh postprocess methods we don't call
    while dumping activations),
  * replaces the sparse attention dispatcher with a plain-SDPA implementation
    (mathematically identical for the var-len batch-1 case we validate),
  * registers a pure-PyTorch submanifold sparse-conv backend under the name
    'none' (gather + GEMM per kernel offset — slow, but it runs anywhere and
    doubles as the executable spec for the C++ implementation).

Import this before importing anything from `trellis2`.
"""

import os
import sys
import types

TRELLIS2_PY = os.environ.get("TRELLIS2_PY", "/trellis2")
if TRELLIS2_PY not in sys.path:
    sys.path.insert(0, TRELLIS2_PY)

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
# The sparse attention config only accepts xformers/flash_attn/flash_attn_3;
# we leave it alone and monkeypatch the dispatcher below instead.

# --- cumesh stub (postprocess-only dependency of representations.mesh.base) ---
if "cumesh" not in sys.modules:
    stub = types.ModuleType("cumesh")

    class _CuMeshStub:
        def __init__(self, *_a, **_k):
            raise RuntimeError("cumesh is stubbed out in the reference container")

    stub.CuMesh = _CuMeshStub
    sys.modules["cumesh"] = stub

# --- o_voxel stub (CUDA hashmap mesher; we dump the decoder's raw 7-channel
# --- output and do mesh comparison with scripts/ref_dual_grid.py instead) ---
if "o_voxel" not in sys.modules:
    ovx = types.ModuleType("o_voxel")
    ovx_convert = types.ModuleType("o_voxel.convert")

    def flexible_dual_grid_to_mesh(*_a, **_k):
        raise RuntimeError("o_voxel is stubbed out in the reference container")

    ovx_convert.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    ovx.convert = ovx_convert
    sys.modules["o_voxel"] = ovx
    sys.modules["o_voxel.convert"] = ovx_convert

# --- flex_gemm stub (CUDA kernels; representations.mesh.base imports
# --- grid_sample_3d for texture baking, unused on the geometry path) ---
if "flex_gemm" not in sys.modules:
    fg = types.ModuleType("flex_gemm")
    fg_ops = types.ModuleType("flex_gemm.ops")
    fg_gs = types.ModuleType("flex_gemm.ops.grid_sample")
    fg_sp = types.ModuleType("flex_gemm.ops.spconv")

    def _fg_unavailable(*_a, **_k):
        raise RuntimeError("flex_gemm is stubbed out in the reference container")

    fg_gs.grid_sample_3d = _fg_unavailable
    fg_sp.sparse_submanifold_conv3d = _fg_unavailable
    fg_ops.grid_sample = fg_gs
    fg_ops.spconv = fg_sp
    fg.ops = fg_ops
    sys.modules["flex_gemm"] = fg
    sys.modules["flex_gemm.ops"] = fg_ops
    sys.modules["flex_gemm.ops.grid_sample"] = fg_gs
    sys.modules["flex_gemm.ops.spconv"] = fg_sp


def _install_sdpa_sparse_attention():
    """Replace trellis2's sparse attention with a dense-SDPA equivalent."""
    import torch
    import torch.nn.functional as F
    from trellis2.modules.sparse.attention import full_attn
    from trellis2.modules.sparse import VarLenTensor

    def sdpa_varlen(q, k, v, q_seqlen, kv_seqlen):
        # q: [Tq, H, C], k/v: [Tkv, H, C] concatenated over batch. Query-chunked
        # so the [H, chunk, L] score matrix stays bounded — the math SDPA backend
        # (forced for true fp32) would otherwise OOM at the cascade's HR token
        # counts. Chunking queries is mathematically exact (each query's softmax
        # is independent), so the golden values are unchanged.
        CHUNK = 2048
        out = torch.empty_like(q)
        qo = ko = 0
        for ql, kl in zip(q_seqlen, kv_seqlen):
            ks = k[ko:ko + kl].transpose(0, 1).unsqueeze(0)  # [1,H,kl,C]
            vs = v[ko:ko + kl].transpose(0, 1).unsqueeze(0)
            for s in range(0, ql, CHUNK):
                e = min(s + CHUNK, ql)
                qs = q[qo + s:qo + e].transpose(0, 1).unsqueeze(0)  # [1,H,chunk,C]
                o = F.scaled_dot_product_attention(qs, ks, vs)
                out[qo + s:qo + e] = o.squeeze(0).transpose(0, 1)
            qo += ql
            ko += kl
        return out

    def sparse_sdpa(*args, **kwargs):
        num = len(args) + len(kwargs)
        if num == 1:
            qkv = args[0] if args else kwargs["qkv"]
            assert isinstance(qkv, VarLenTensor)
            q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
            q, k, v = qkv.feats.unbind(dim=1)  # [T,3,H,C] -> 3x[T,H,C]
            out = sdpa_varlen(q, k, v, q_seqlen, q_seqlen)
            return qkv.replace(out)
        if num == 2:
            q = args[0] if len(args) > 0 else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
                qf = q.feats
            else:
                N, L = q.shape[:2]
                q_seqlen = [L] * N
                qf = q.reshape(N * L, *q.shape[2:])
            if isinstance(kv, VarLenTensor):
                kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
                kvf = kv.feats
            else:
                N, L = kv.shape[:2]
                kv_seqlen = [L] * N
                kvf = kv.reshape(N * L, *kv.shape[2:])
            k, v = kvf.unbind(dim=1)
            out = sdpa_varlen(qf, k, v, q_seqlen, kv_seqlen)
            if s is not None:
                return s.replace(out)
            N = len(q_seqlen)
            return out.reshape(N, q_seqlen[0], *out.shape[1:])
        if num == 3:
            q = args[0] if len(args) > 0 else kwargs["q"]
            k = args[1] if len(args) > 1 else kwargs["k"]
            v = args[2] if len(args) > 2 else kwargs["v"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
                qf = q.feats
            else:
                N, L = q.shape[:2]
                q_seqlen = [L] * N
                qf = q.reshape(N * L, *q.shape[2:])
            if isinstance(k, VarLenTensor):
                kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
                kf, vf = k.feats, v.feats
            else:
                N, L = k.shape[:2]
                kv_seqlen = [L] * N
                kf = k.reshape(N * L, *k.shape[2:])
                vf = v.reshape(N * L, *v.shape[2:])
            out = sdpa_varlen(qf, kf, vf, q_seqlen, kv_seqlen)
            if s is not None:
                return s.replace(out)
            N = len(q_seqlen)
            return out.reshape(N, q_seqlen[0], *out.shape[1:])
        raise AssertionError("bad arg count")

    full_attn.sparse_scaled_dot_product_attention = sparse_sdpa
    # modules.py imported the symbol by value; patch it there too.
    from trellis2.modules.sparse.attention import modules as attn_modules
    attn_modules.sparse_scaled_dot_product_attention = sparse_sdpa


def _install_torch_sparse_conv():
    """Register a pure-PyTorch submanifold conv backend as 'none'."""
    import math
    import torch
    import torch.nn as nn

    from trellis2.modules.sparse.conv import conv as conv_dispatch

    mod = types.ModuleType("trellis2.modules.sparse.conv.conv_none")

    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size,
                           stride=1, dilation=1, padding=None, bias=True,
                           indice_key=None):
        assert stride == 1 and padding is None, "submanifold only"
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.kernel_size = ks
        self.stride = (1, 1, 1)
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        # flex_gemm weight layout: (Co, Kd, Kh, Kw, Ci)
        self.weight = nn.Parameter(torch.empty(out_channels, *ks, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _coord_key(coords, spatial_shape):
        # coords: [N,4] int (b,x,y,z) -> int64 linear key
        b, x, y, z = coords.unbind(-1)
        sx, sy, sz = spatial_shape
        return ((b.long() * sx + x.long()) * sy + y.long()) * sz + z.long()

    def sparse_conv3d_forward(self, x):
        coords = x.coords
        feats = x.feats
        n = feats.shape[0]
        spatial = tuple(x.spatial_shape)
        keys = _coord_key(coords, spatial)
        order = torch.argsort(keys)
        keys_sorted = keys[order]

        Co, Kd, Kh, Kw, Ci = self.weight.shape
        w = self.weight
        out = feats.new_zeros(n, Co)
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        rd, rh, rw = Kd // 2, Kh // 2, Kw // 2
        dd, dh, dw = self.dilation
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    off = coords.new_tensor([0, (kd - rd) * dd, (kh - rh) * dh, (kw - rw) * dw])
                    ncoords = coords + off
                    inb = ((ncoords[:, 1] >= 0) & (ncoords[:, 1] < spatial[0]) &
                           (ncoords[:, 2] >= 0) & (ncoords[:, 2] < spatial[1]) &
                           (ncoords[:, 3] >= 0) & (ncoords[:, 3] < spatial[2]))
                    nkeys = _coord_key(ncoords, spatial)
                    pos = torch.searchsorted(keys_sorted, nkeys)
                    pos_c = pos.clamp(max=n - 1)
                    hit = inb & (keys_sorted[pos_c] == nkeys)
                    src = order[pos_c[hit]]
                    # out[i] += feats[neighbor(i, offset)] @ w[:, kd, kh, kw, :]^T
                    contrib = feats[src] @ w[:, kd, kh, kw, :].to(feats.dtype).t()
                    out[hit] += contrib
        return x.replace(out)

    def sparse_inverse_conv3d_init(self, *a, **k):
        raise NotImplementedError

    def sparse_inverse_conv3d_forward(self, x):
        raise NotImplementedError

    mod.sparse_conv3d_init = sparse_conv3d_init
    mod.sparse_conv3d_forward = sparse_conv3d_forward
    mod.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
    mod.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
    conv_dispatch._backends["none"] = mod
    from trellis2.modules import sparse as sp
    sp.config.CONV = "none"


def _force_true_fp32():
    """Make CUDA math bit-comparable to a real fp32 reference.

    PyTorch's default CUDA matmul/attention uses TF32 (≈10-bit mantissa) and
    flash/mem-efficient SDPA kernels that accumulate in reduced precision —
    that shows up as ~1e-3 relative error versus a true fp32 (or ggml-CPU)
    forward, which would otherwise masquerade as a port bug. Force full-width
    fp32 so the golden dumps are the real reference regardless of --device.
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _redirect_pipeline_log():
    """Point the reference tree's pipeline logger at a writable location.

    Some TRELLIS.2 checkouts log to a hardcoded POSIX path ("/tmp/..."). On
    Windows that resolves against the current drive (F:\\tmp\\...), which
    normally does not exist, so the decoder dies inside a logging FileHandler
    instead of on anything numerical. Only redirected when the original
    directory is missing, so the container — where /tmp exists — is untouched.
    Checkouts without the module simply have nothing to patch.
    """
    import importlib
    import tempfile

    try:
        mod = importlib.import_module("trellis2.utils.pipeline_logger")
    except Exception:
        return
    path = getattr(mod, "LOG_PATH", None)
    if not isinstance(path, str) or not path:
        return
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        mod.LOG_PATH = os.path.join(tempfile.gettempdir(), os.path.basename(path))


def setup():
    _force_true_fp32()
    _install_sdpa_sparse_attention()
    _install_torch_sparse_conv()
    _redirect_pipeline_log()


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")
DUMPS = os.path.join(REPO, "dumps")


def write_dinodata(path, arr):
    """.dinodata: DINOCOND | u32 version | u32 dtype(0=f32) | u32 ndim | dims | f32 payload"""
    import struct
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"DINOCOND")
        f.write(struct.pack("<III", 1, 0, arr.ndim))
        f.write(struct.pack("<%dI" % arr.ndim, *arr.shape))
        f.write(arr.tobytes())


def preprocess_rgba(img):
    """pipeline.preprocess_image for an RGBA input (no rembg): downscale to
    <=1024, alpha bbox square crop, premultiply onto black. Returns RGB PIL."""
    import numpy as np
    from PIL import Image

    assert img.mode == "RGBA", "fixture must have an alpha channel"
    max_size = max(img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.Resampling.LANCZOS)
    out = np.array(img)
    alpha = out[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    img = img.crop(bbox)
    out = np.array(img).astype(np.float32) / 255
    out = out[:, :, :3] * out[:, :, 3:4]
    return Image.fromarray((out * 255).astype(np.uint8))
