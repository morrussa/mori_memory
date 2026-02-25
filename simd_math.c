#include <immintrin.h> // AVX, FMA
#include <math.h>
#include <stdint.h>

float cosine_similarity_avx(const float* v1, const float* v2, size_t n) {
    // 1. 极小数组标量兜底
    if (n < 8) {
        float dot = 0, norm1 = 0, norm2 = 0;
        for (size_t i = 0; i < n; ++i) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 == 0 || norm2 == 0) return 0.0f;
        return dot / (sqrtf(norm1) * sqrtf(norm2));
    }

    // 2. 初始化累加器 (4路展开，每路处理8个float，共32个/循环)
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
    size_t limit = n - (n % 32); // 每次处理 32 个 float

    for (; i < limit; i += 32) {
        // 正确的加载方式：每次偏移 8 个 float
        __m256 a0 = _mm256_loadu_ps(v1 + i);      // 索引 0-7
        __m256 b0 = _mm256_loadu_ps(v2 + i);
        
        __m256 a1 = _mm256_loadu_ps(v1 + i + 8);  // 索引 8-15
        __m256 b1 = _mm256_loadu_ps(v2 + i + 8);
        
        __m256 a2 = _mm256_loadu_ps(v1 + i + 16); // 索引 16-23
        __m256 b2 = _mm256_loadu_ps(v2 + i + 16);
        
        __m256 a3 = _mm256_loadu_ps(v1 + i + 24); // 索引 24-31
        __m256 b3 = _mm256_loadu_ps(v2 + i + 24);

        // FMA 计算
        sum_dot0 = _mm256_fmadd_ps(a0, b0, sum_dot0);
        sum_n10  = _mm256_fmadd_ps(a0, a0, sum_n10);
        sum_n20  = _mm256_fmadd_ps(b0, b0, sum_n20);

        sum_dot1 = _mm256_fmadd_ps(a1, b1, sum_dot1);
        sum_n11  = _mm256_fmadd_ps(a1, a1, sum_n11);
        sum_n21  = _mm256_fmadd_ps(b1, b1, sum_n21);

        sum_dot2 = _mm256_fmadd_ps(a2, b2, sum_dot2);
        sum_n12  = _mm256_fmadd_ps(a2, a2, sum_n12);
        sum_n22  = _mm256_fmadd_ps(b2, b2, sum_n22);

        sum_dot3 = _mm256_fmadd_ps(a3, b3, sum_dot3);
        sum_n13  = _mm256_fmadd_ps(a3, a3, sum_n13);
        sum_n23  = _mm256_fmadd_ps(b3, b3, sum_n23);
    }

    // 3. 合并累加器
    __m256 sum_dot = _mm256_add_ps(_mm256_add_ps(sum_dot0, sum_dot1), _mm256_add_ps(sum_dot2, sum_dot3));
    __m256 sum_n1  = _mm256_add_ps(_mm256_add_ps(sum_n10,  sum_n11),  _mm256_add_ps(sum_n12, sum_n13));
    __m256 sum_n2  = _mm256_add_ps(_mm256_add_ps(sum_n20,  sum_n21),  _mm256_add_ps(sum_n22, sum_n23));

    // 4. 处理剩余元素 (32的倍数之后剩余的部分)
    for (; i < n; ++i) {
        sum_dot[0] += v1[i] * v2[i];
    }


    float dot_rem = 0.0f, n1_rem = 0.0f, n2_rem = 0.0f;
    for (; i < n; ++i) {
        dot_rem += v1[i] * v2[i];
        n1_rem  += v1[i] * v1[i];
        n2_rem  += v2[i] * v2[i];
    }

    // 5. 水平求和
    // 先将 256 位向量相加：[S0..S7] + [S8..S15] (如果还有没合并的)
    // 这里我们已经合并到了 sum_dot 中
    
    // low 128-bit + high 128-bit
    __m128 sum_dot_low = _mm_add_ps(_mm256_castps256_ps128(sum_dot), _mm256_extractf128_ps(sum_dot, 1));
    __m128 sum_n1_low  = _mm_add_ps(_mm256_castps256_ps128(sum_n1),  _mm256_extractf128_ps(sum_n1, 1));
    __m128 sum_n2_low  = _mm_add_ps(_mm256_castps256_ps128(sum_n2),  _mm256_extractf128_ps(sum_n2, 1));

    // 再将 128 位的高低位互换并相加
    __m128 sum_dot_high = _mm_movehl_ps(sum_dot_low, sum_dot_low);
    __m128 sum_n1_high  = _mm_movehl_ps(sum_n1_low, sum_n1_low);
    __m128 sum_n2_high  = _mm_movehl_ps(sum_n2_low, sum_n2_low);

    // 再相加得到最终两个数，再进行一次水平加 (或者直接加标量)
    __m128 sum_dot_final = _mm_add_ps(sum_dot_low, sum_dot_high); // [S0+S2, S1+S3, ...]
    __m128 sum_n1_final  = _mm_add_ps(sum_n1_low, sum_n1_high);
    __m128 sum_n2_final  = _mm_add_ps(sum_n2_low, sum_n2_high);
    
    __m128 shuf = _mm_shuffle_ps(sum_dot_final, sum_dot_final, _MM_SHUFFLE(1, 1, 1, 1)); // 取第二个元素
    float dot = _mm_cvtss_f32(_mm_add_ss(sum_dot_final, shuf)); // 第一个元素 + 第二个元素

    shuf = _mm_shuffle_ps(sum_n1_final, sum_n1_final, _MM_SHUFFLE(1, 1, 1, 1));
    float norm1 = _mm_cvtss_f32(_mm_add_ss(sum_n1_final, shuf));

    shuf = _mm_shuffle_ps(sum_n2_final, sum_n2_final, _MM_SHUFFLE(1, 1, 1, 1));
    float norm2 = _mm_cvtss_f32(_mm_add_ss(sum_n2_final, shuf));

    // 加上标量尾部
    dot += dot_rem;
    norm1 += n1_rem;
    norm2 += n2_rem;

    if (norm1 == 0 || norm2 == 0) return 0.0f;
    return dot / (sqrtf(norm1) * sqrtf(norm2));
}