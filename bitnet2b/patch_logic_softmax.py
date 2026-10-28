#!/usr/bin/env python3
from pathlib import Path
import math, re, sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "BitNet")
p = root / "3rdparty/llama.cpp/ggml/src/ggml-cpu/vec.cpp"
s = p.read_text()

vals = []
for i in range(1025):
    x = f"{math.exp(-i/64):.9g}"
    if "e" not in x and "." not in x:
        x += ".0"
    vals.append(x + "f")

lut = """
static const float llf_exp_gap_lut[1025] = {
""" + ", ".join(vals) + """
};

static inline float llf_exp_gap_scalar(float x, float maxv) {
    float gap = maxv - x;
    if (gap <= 0.0f) return 1.0f;
    if (gap >= 16.0f) return 0.0f;
    int idx = (int)(gap * 64.0f + 0.5f);
    if (idx < 0) idx = 0;
    if (idx > 1024) idx = 1024;
    return llf_exp_gap_lut[idx];
}
"""

pat = re.compile(
    r"ggml_float ggml_vec_soft_max_f32\(const int n, float \* y, const float \* x, float max\) \{.*?\n\}\n\n"
    r"(?=ggml_float ggml_vec_log_soft_max_f32)",
    re.S,
)
m = pat.search(s)
if not m:
    raise SystemExit("softmax function not found")

fn = r"""ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;

#if defined(__AVX512F__)
    const __m512 vmax = _mm512_set1_ps(max);
    const __m512 vscale = _mm512_set1_ps(64.0f);
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vsixteen = _mm512_set1_ps(16.0f);
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 gap = _mm512_sub_ps(vmax, vx);
        gap = _mm512_max_ps(vzero, _mm512_min_ps(gap, vsixteen));
        __m512i idx = _mm512_cvtps_epi32(_mm512_mul_ps(gap, vscale));
        __m512 val = _mm512_i32gather_ps(idx, llf_exp_gap_lut, 4);
        __mmask16 keep = _mm512_cmp_ps_mask(gap, vsixteen, _CMP_LT_OQ);
        val = _mm512_maskz_mov_ps(keep, val);
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__)
    const __m256 vmax = _mm256_set1_ps(max);
    const __m256 vscale = _mm256_set1_ps(64.0f);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vsixteen = _mm256_set1_ps(16.0f);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 gap = _mm256_sub_ps(vmax, vx);
        gap = _mm256_max_ps(vzero, _mm256_min_ps(gap, vsixteen));
        __m256i idx = _mm256_cvtps_epi32(_mm256_mul_ps(gap, vscale));
        __m256 val = _mm256_i32gather_ps(llf_exp_gap_lut, idx, 4);
        __m256 keep = _mm256_cmp_ps(gap, vsixteen, _CMP_LT_OQ);
        val = _mm256_and_ps(val, keep);
        _mm256_storeu_ps(y + i, val);
        __m128 lo = _mm256_castps256_ps128(val);
        __m128 hi = _mm256_extractf128_ps(val, 1);
        __m128 z = _mm_add_ps(lo, hi);
        z = _mm_hadd_ps(z, z);
        z = _mm_hadd_ps(z, z);
        sum += (ggml_float)_mm_cvtss_f32(z);
    }
#endif

    for (; i < n; ++i) {
        const float val = llf_exp_gap_scalar(x[i], max);
        y[i] = val;
        sum += (ggml_float) val;
    }
    return sum;
}

"""

s = s[:m.start()] + lut + "\n" + fn + s[m.end():]
p.write_text(s)
print("patched", p)
