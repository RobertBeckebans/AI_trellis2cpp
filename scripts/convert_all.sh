#!/usr/bin/env bash
# Convert every stage-1 geometry checkpoint to GGUF (f16 for the demo, f32 for
# validation). Run inside the reference container:
#   docker run --rm -v "$PWD":/work -w /work trellis2-ref bash scripts/convert_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p ggufs

T2=models/TRELLIS.2-4B/ckpts
T1=models/TRELLIS-image-large/ckpts
PJ=models/TRELLIS.2-4B/pipeline.json
TPJ=models/TRELLIS.2-4B/texturing_pipeline.json

for ft in 0 1; do
    suf=$([ "$ft" = 0 ] && echo f32 || echo f16)
    python scripts/convert_dino_to_gguf.py                                       --output ggufs/dino_$suf.gguf      --ftype "$ft"
    python scripts/convert_ss_flow_to_gguf.py   --model $T2/ss_flow_img_dit_1_3B_64_bf16.safetensors            --output ggufs/ss_flow_$suf.gguf   --ftype "$ft"
    python scripts/convert_ss_dec_to_gguf.py    --model $T1/ss_dec_conv3d_16l8_fp16.safetensors                 --output ggufs/ss_dec_$suf.gguf    --ftype "$ft"
    python scripts/convert_slat_flow_to_gguf.py --model $T2/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors  --pipeline-json $PJ --output ggufs/slat_flow_$suf.gguf      --ftype "$ft"
    python scripts/convert_slat_flow_to_gguf.py --model $T2/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors --pipeline-json $PJ --output ggufs/slat_flow_1024_$suf.gguf --ftype "$ft"
    python scripts/convert_shape_dec_to_gguf.py --model $T2/shape_dec_next_dc_f16c32_fp16.safetensors           --output ggufs/shape_dec_$suf.gguf --ftype "$ft"
    # PBR texturing stack (shape encoder retained for standalone parity tooling)
    python scripts/convert_tex_dec_to_gguf.py   --model $T2/tex_dec_next_dc_f16c32_fp16.safetensors             --output ggufs/tex_dec_$suf.gguf   --ftype "$ft"
    python scripts/convert_shape_enc_to_gguf.py --model $T2/shape_enc_next_dc_f16c32_fp16.safetensors           --output ggufs/shape_enc_$suf.gguf --ftype "$ft"
    python scripts/convert_tex_flow_to_gguf.py  --model $T2/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors  --texturing-json $TPJ --output ggufs/tex_slat_flow_512_$suf.gguf  --ftype "$ft"
    python scripts/convert_tex_flow_to_gguf.py  --model $T2/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors --texturing-json $TPJ --output ggufs/tex_slat_flow_1024_$suf.gguf --ftype "$ft"
done

echo "all GGUFs written:"
ls -la ggufs/
