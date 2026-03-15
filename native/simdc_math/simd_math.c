#include <immintrin.h> // AVX, FMA
#include <stddef.h>
#include <math.h>
#include <stdint.h>

static inline float hsum256_ps(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

float dot_product_avx(const float* v1, const float* v2, size_t n) {
    if (n < 8) {
        float dot = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            dot += v1[i] * v2[i];
        }
        return dot;
    }

    __m256 sum_dot0 = _mm256_setzero_ps();
    __m256 sum_dot1 = _mm256_setzero_ps();
    __m256 sum_dot2 = _mm256_setzero_ps();
    __m256 sum_dot3 = _mm256_setzero_ps();

    size_t i = 0;
    size_t limit = n - (n % 32);

    for (; i < limit; i += 32) {
        if (i + 128 < n) {
            _mm_prefetch((const char*)(v1 + i + 128), _MM_HINT_T0);
            _mm_prefetch((const char*)(v2 + i + 128), _MM_HINT_T0);
        }

        __m256 a0 = _mm256_loadu_ps(v1 + i);
        __m256 b0 = _mm256_loadu_ps(v2 + i);
        __m256 a1 = _mm256_loadu_ps(v1 + i + 8);
        __m256 b1 = _mm256_loadu_ps(v2 + i + 8);
        __m256 a2 = _mm256_loadu_ps(v1 + i + 16);
        __m256 b2 = _mm256_loadu_ps(v2 + i + 16);
        __m256 a3 = _mm256_loadu_ps(v1 + i + 24);
        __m256 b3 = _mm256_loadu_ps(v2 + i + 24);

        sum_dot0 = _mm256_fmadd_ps(a0, b0, sum_dot0);
        sum_dot1 = _mm256_fmadd_ps(a1, b1, sum_dot1);
        sum_dot2 = _mm256_fmadd_ps(a2, b2, sum_dot2);
        sum_dot3 = _mm256_fmadd_ps(a3, b3, sum_dot3);
    }

    __m256 sum_dot = _mm256_add_ps(_mm256_add_ps(sum_dot0, sum_dot1), _mm256_add_ps(sum_dot2, sum_dot3));

    float dot = hsum256_ps(sum_dot);
    for (; i < n; ++i) {
        dot += v1[i] * v2[i];
    }
    return dot;
}

float cosine_similarity_avx(const float* v1, const float* v2, size_t n) {
    if (n < 8) {
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
        return dot / (sqrtf(norm1) * sqrtf(norm2));
    }

    __m256 sum_dot0 = _mm256_setzero_ps();
    __m256 sum_dot1 = _mm256_setzero_ps();
    __m256 sum_dot2 = _mm256_setzero_ps();
    __m256 sum_dot3 = _mm256_setzero_ps();

    __m256 sum_n10 = _mm256_setzero_ps();
    __m256 sum_n11 = _mm256_setzero_ps();
    __m256 sum_n12 = _mm256_setzero_ps();
    __m256 sum_n13 = _mm256_setzero_ps();

    __m256 sum_n20 = _mm256_setzero_ps();
    __m256 sum_n21 = _mm256_setzero_ps();
    __m256 sum_n22 = _mm256_setzero_ps();
    __m256 sum_n23 = _mm256_setzero_ps();

    size_t i = 0;
    size_t limit = n - (n % 32);

    for (; i < limit; i += 32) {
        if (i + 128 < n) {
            _mm_prefetch((const char*)(v1 + i + 128), _MM_HINT_T0);
            _mm_prefetch((const char*)(v2 + i + 128), _MM_HINT_T0);
        }

        __m256 a0 = _mm256_loadu_ps(v1 + i);
        __m256 b0 = _mm256_loadu_ps(v2 + i);
        __m256 a1 = _mm256_loadu_ps(v1 + i + 8);
        __m256 b1 = _mm256_loadu_ps(v2 + i + 8);
        __m256 a2 = _mm256_loadu_ps(v1 + i + 16);
        __m256 b2 = _mm256_loadu_ps(v2 + i + 16);
        __m256 a3 = _mm256_loadu_ps(v1 + i + 24);
        __m256 b3 = _mm256_loadu_ps(v2 + i + 24);

        sum_dot0 = _mm256_fmadd_ps(a0, b0, sum_dot0);
        sum_n10 = _mm256_fmadd_ps(a0, a0, sum_n10);
        sum_n20 = _mm256_fmadd_ps(b0, b0, sum_n20);

        sum_dot1 = _mm256_fmadd_ps(a1, b1, sum_dot1);
        sum_n11 = _mm256_fmadd_ps(a1, a1, sum_n11);
        sum_n21 = _mm256_fmadd_ps(b1, b1, sum_n21);

        sum_dot2 = _mm256_fmadd_ps(a2, b2, sum_dot2);
        sum_n12 = _mm256_fmadd_ps(a2, a2, sum_n12);
        sum_n22 = _mm256_fmadd_ps(b2, b2, sum_n22);

        sum_dot3 = _mm256_fmadd_ps(a3, b3, sum_dot3);
        sum_n13 = _mm256_fmadd_ps(a3, a3, sum_n13);
        sum_n23 = _mm256_fmadd_ps(b3, b3, sum_n23);
    }

    __m256 sum_dot = _mm256_add_ps(_mm256_add_ps(sum_dot0, sum_dot1), _mm256_add_ps(sum_dot2, sum_dot3));
    __m256 sum_n1 = _mm256_add_ps(_mm256_add_ps(sum_n10, sum_n11), _mm256_add_ps(sum_n12, sum_n13));
    __m256 sum_n2 = _mm256_add_ps(_mm256_add_ps(sum_n20, sum_n21), _mm256_add_ps(sum_n22, sum_n23));

    float dot_rem = 0.0f, n1_rem = 0.0f, n2_rem = 0.0f;
    for (; i < n; ++i) {
        dot_rem += v1[i] * v2[i];
        n1_rem += v1[i] * v1[i];
        n2_rem += v2[i] * v2[i];
    }

    float dot = hsum256_ps(sum_dot) + dot_rem;
    float norm1 = hsum256_ps(sum_n1) + n1_rem;
    float norm2 = hsum256_ps(sum_n2) + n2_rem;

    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    return dot / (sqrtf(norm1) * sqrtf(norm2));
}
