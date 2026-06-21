#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef SIMD
#include <arm_neon.h>
#endif

#include "common.h"
#include "compute.h"
#include <linux/time.h>

void zero_z() {
  for (int i = 0; i != m; ++i) {
    for (int j = 0; j != n; ++j) {
      Z[i][j] = 0;
    }
  }
}

void compute_row_major_mnk() {
  zero_z();
  for (int i = 0; i != m; ++i) {
    for (int j = 0; j != n; ++j) {
      for (int l = 0; l != k; ++l) {
        Z[i][j] += X[i][l] * Y[l][j];
      }
    }
  }
}

void compute_row_major_mkn() {
  // TODO: task 1
  zero_z();
  for (int i = 0; i != m; ++i) {
    for (int l = 0; l != k; ++l) {
      for (int j = 0; j != n; ++j) {
        Z[i][j] += X[i][l] * Y[l][j];
      }
    }
  }
}

void compute_row_major_kmn() {
  // TODO: task 1
}

void compute_row_major_nmk() {
  // TODO: task 1
}

void compute_row_major_nkm() {
  // TODO: task 1
}

void compute_row_major_knm() {
  // TODO: task 1
}

void compute_y_transpose_mnk() {
  // TODO: task 2
  zero_z();
  for (int i = 0; i != m; ++i) {
    for (int j = 0; j != n; ++j) {
      for (int l = 0; l != k; ++l) {
        Z[i][j] += X[i][l] * YP[j][l];
      }
    }
  }
}

void compute_b16() {
  // TODO: task 2
  const int B = 16;
  zero_z();
  for (int i = 0; i < m; i += B) {
    for (int j = 0; j < n; j += B) {
      for (int l = 0; l < k; l += B) {
        // kmn order
        for (int ll = l; ll < l + B && ll < k; ++ll)
          for (int ii = i; ii < i + B && ii < m; ++ii)
            for (int jj = j; jj < j + B && jj < n; ++jj) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
            }
      }
    }
  }
}

void compute_row_major_mnkkmn_b32() {
  // TODO: task 2
  const int B = 32;
  zero_z();
  for (int i = 0; i < m; i += B) {
    for (int j = 0; j < n; j += B) {
      for (int l = 0; l < k; l += B) {
        // kmn order
        for (int ll = l; ll < l + B && ll < k; ++ll)
          for (int ii = i; ii < i + B && ii < m; ++ii)
            for (int jj = j; jj < j + B && jj < n; ++jj) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
            }
      }
    }
  }
}

void compute_row_major_mnk_lu2() {
  // TODO: task 2
  zero_z();
  if (k & 1) { // even ver
    for (int i = 0; i != m; ++i) {
      for (int j = 0; j != n; ++j) {
        int l = 0;
        for (; l < k - 1; l += 2) {
          Z[i][j] += X[i][l] * Y[l][j];
          Z[i][j] += X[i][l + 1] * Y[l + 1][j];
        }
        Z[i][j] += X[i][l] * Y[l][j];
      }
    }
  } else {
    for (int i = 0; i != m; ++i) {
      for (int j = 0; j != n; ++j) {
        for (int l = 0; l != k; l += 2) {
          Z[i][j] += X[i][l] * Y[l][j];
          Z[i][j] += X[i][l + 1] * Y[l + 1][j];
        }
      }
    }
  }
}

void compute_mknmkn_b32_lu8() {
  const int B = 32;
  zero_z();
  for (int i = 0; i < m; i += B) {
    int imax = i + B < m ? i + B : m;
    for (int l = 0; l < k; l += B) {
      int lmax = l + B < k ? l + B : k;
      for (int j = 0; j < n; j += B) {

        int jmax = ((j + B < n) ? j + B : n) - (8 - 1);
        for (int ii = i; ii < imax; ++ii)
          for (int ll = l; ll < lmax; ++ll) {
            for (int jj = j; jj < jmax; jj += 8) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
              Z[ii][jj + 1] += X[ii][ll] * Y[ll][jj + 1];
              Z[ii][jj + 2] += X[ii][ll] * Y[ll][jj + 2];
              Z[ii][jj + 3] += X[ii][ll] * Y[ll][jj + 3];
              Z[ii][jj + 4] += X[ii][ll] * Y[ll][jj + 4];
              Z[ii][jj + 5] += X[ii][ll] * Y[ll][jj + 5];
              Z[ii][jj + 6] += X[ii][ll] * Y[ll][jj + 6];
              Z[ii][jj + 7] += X[ii][ll] * Y[ll][jj + 7];
            }
            // handle remaining columns
            for (int jj = jmax; jj < j + B && jj < n; ++jj) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
            }
          }
      }
    }
  }
}

void compute_mknmkn_b32_Bj64_lu8() {
  const int B = 32;
  zero_z();
  for (int i = 0; i < m; i += B) {
    int imax = i + B < m ? i + B : m;
    for (int l = 0; l < k; l += B) {
      int lmax = l + B < k ? l + B : k;
      for (int j = 0; j < n; j += 64) {

        int jmax = ((j + 64 < n) ? j + 64 : n) - (8 - 1);
        for (int ii = i; ii < imax; ++ii)
          for (int ll = l; ll < lmax; ++ll) {
            for (int jj = j; jj < jmax; jj += 8) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
              Z[ii][jj + 1] += X[ii][ll] * Y[ll][jj + 1];
              Z[ii][jj + 2] += X[ii][ll] * Y[ll][jj + 2];
              Z[ii][jj + 3] += X[ii][ll] * Y[ll][jj + 3];
              Z[ii][jj + 4] += X[ii][ll] * Y[ll][jj + 4];
              Z[ii][jj + 5] += X[ii][ll] * Y[ll][jj + 5];
              Z[ii][jj + 6] += X[ii][ll] * Y[ll][jj + 6];
              Z[ii][jj + 7] += X[ii][ll] * Y[ll][jj + 7];
            }
            // handle remaining columns
            for (int jj = jmax; jj < j + 64 && jj < n; ++jj) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
            }
          }
      }
    }
  }
}

void compute_mknmkn_b32_Bj128_lu8() {
  const int B = 32;
  zero_z();
  for (int i = 0; i < m; i += B) {
    int imax = i + B < m ? i + B : m;
    for (int l = 0; l < k; l += B) {
      int lmax = l + B < k ? l + B : k;
      for (int j = 0; j < n; j += 128) {

        int jmax = ((j + 128 < n) ? j + 128 : n) - (8 - 1);
        for (int ii = i; ii < imax; ++ii)
          for (int ll = l; ll < lmax; ++ll) {
            for (int jj = j; jj < jmax; jj += 8) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
              Z[ii][jj + 1] += X[ii][ll] * Y[ll][jj + 1];
              Z[ii][jj + 2] += X[ii][ll] * Y[ll][jj + 2];
              Z[ii][jj + 3] += X[ii][ll] * Y[ll][jj + 3];
              Z[ii][jj + 4] += X[ii][ll] * Y[ll][jj + 4];
              Z[ii][jj + 5] += X[ii][ll] * Y[ll][jj + 5];
              Z[ii][jj + 6] += X[ii][ll] * Y[ll][jj + 6];
              Z[ii][jj + 7] += X[ii][ll] * Y[ll][jj + 7];
            }
            // handle remaining columns
            for (int jj = jmax; jj < j + 128 && jj < n; ++jj) {
              Z[ii][jj] += X[ii][ll] * Y[ll][jj];
            }
          }
      }
    }
  }
}

void compute_simd() {
#ifdef SIMD
  // TODO: task 3
  const int B = 64, quar = 4;
  zero_z();
  for (int i = 0; i != m; ++i)
    for (int j = 0; j != n; ++j) {
      uint64x2_t sum = vdupq_n_u64(0);
      for (int l = 0; l + 3 < k; l += 4) {
        // loaded YP16[][]
        uint16x4_t y_vec = vld1_u16(&YP16[j][l]);
        uint16x4_t x_vec = vld1_u16(&X16[i][l]);

        // 计算 32 位乘法结果
        uint32x4_t mul = vmull_u16(x_vec, y_vec);
        // 拆分为两个 64 位扩展并累加
        uint64x2_t ext_low = vmovl_u32(vget_low_u32(mul));
        uint64x2_t ext_high = vmovl_u32(vget_high_u32(mul));

        sum = vaddq_u64(sum, ext_low);
        sum = vaddq_u64(sum, ext_high);
      }
      Z[i][j] += vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);

      for (int l = k - (k % 4); l < k; ++l) {
        Z[i][j] += (uint32_t)X16[i][l] * YP16[j][l];
      }
    }
  /* batched version, not faster than not batched
  for(int i=0; i < m; i+=B) {
      for(int j=0; j < n; j+=B) {
          for(int l=0; l < k; l+=B) {
              for(int ii = i; ii < i+B && ii < m; ++ii)
               for(int jj = j; jj < j+B && jj < n; ++jj){
                  int mx = ((l+B < k) ? l+B : k) - (quar - 1), ll = l;
                  uint64x2_t low = vdupq_n_u64(0);
                  uint64x2_t high = vdupq_n_u64(0);
                  for(; ll < mx; ll += quar) {
                      // loaded YP16[][]
                      uint16x4_t y_vec = vld1_u16(&YP16[jj][ll]);
                      uint16x4_t x_vec = vld1_u16(&X16[ii][ll]);

                      // 计算 32 位乘法结果
                      uint32x4_t mul = vmull_u16(x_vec, y_vec);

                      // 拆分为两个 64 位扩展并累加
                      uint64x2_t ext_low = vmovl_u32(vget_low_u32(mul));
                      uint64x2_t ext_high = vmovl_u32(vget_high_u32(mul));

                      low = vaddq_u64(low, ext_low);
                      high = vaddq_u64(high, ext_high);
                  }
                  uint64_t vector_sum = vgetq_lane_u64(low, 0) +
  vgetq_lane_u64(low, 1) + vgetq_lane_u64(high, 0) + vgetq_lane_u64(high,
  1); Z[ii][jj] += vector_sum; for(; ll < l + B && ll < k; ++ll) {
                      Z[ii][jj] += (uint32_t)X16[ii][ll] * YP16[jj][ll];
                  }
               }
          }
      }
  }
  */
#endif
}
void simd_ver2() {
#ifdef SIMD
  // load X[][], Y[][] in natural way
  const int B = 32;
  const int lu = 4;
  zero_z();
  for (int i = 0; i < m; i += B) {
    for (int l = 0; l < k; l += B) {
      for (int j = 0; j < n; j += B) {
        // mkn again
        int imax = i + B < m ? i + B : m;
        int lmax = l + B < k ? l + B : k;
        int jmax = ((j + B < n) ? j + B : n) - (lu - 1);
        for (int ii = i; ii != imax; ++ii) {
          // n is always odd
          for (int jj = j; jj < jmax - 1; jj += 2) {
            uint64x2_t sum = vld1q_u64(&Z[ii][jj]);
            // loaded Y32 in natural way
            for (int ll = l; ll < lmax; ++ll) {
              uint32x2_t y_vec = vld1_u32(&Y32[ll][jj]);
              sum = vmlsl_n_u32(sum, y_vec, X32[ii][ll]);
            }
            // load sum back to Z
            vst1q_u64(&Z[ii][jj], sum);
          }
        }
      }
    }
  }
#endif
}

uint64_t elapsed(const struct timespec start, const struct timespec end) {
  struct timespec result;
  result.tv_sec = end.tv_sec - start.tv_sec;
  result.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (result.tv_nsec < 0) {
    --result.tv_sec;
    result.tv_nsec += SEC;
  }
  uint64_t res = result.tv_sec * SEC + result.tv_nsec;
  return res;
}

uint64_t compute() {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  switch (COMPUTE_SELECT) {
  case COMPUTE_ROW_MAJOR_MNK:
    // printf("COMPUTE_ROW_MAJOR_MNK\n");
    compute_row_major_mnk();
    break;
  case COMPUTE_ROW_MAJOR_MKN:
    // printf("COMPUTE_ROW_MAJOR_MKN\n");
    compute_row_major_mkn();
    break;
  case COMPUTE_ROW_MAJOR_KMN:
    // printf("COMPUTE_ROW_MAJOR_KMN\n");
    compute_row_major_kmn();
    break;
  case COMPUTE_ROW_MAJOR_NMK:
    // printf("COMPUTE_ROW_MAJOR_NMK\n");
    compute_row_major_nmk();
    break;
  case COMPUTE_ROW_MAJOR_NKM:
    // printf("COMPUTE_ROW_MAJOR_NKM\n");
    compute_row_major_nkm();
    break;
  case COMPUTE_ROW_MAJOR_KNM:
    // printf("COMPUTE_ROW_MAJOR_KNM\n");
    compute_row_major_knm();
    break;
  case COMPUTE_Y_TRANSPOSE_MNK:
    // printf("COMPUTE_Y_TRANSPOSE_MNK\n");
    compute_y_transpose_mnk();
    break;
  case COMPUTE_ROW_MAJOR_MNKKMN_B32:
    // printf("COMPUTE_ROW_MAJOR_MNKKMN_B32\n");
    compute_row_major_mnkkmn_b32();
    break;
  case COMPUTE_ROW_MAJOR_MNK_LU2:
    // printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
    compute_row_major_mnk_lu2();
    break;
  case COMPUTE_SIMD:
    // printf("COMPUTE_SIMD\n");
    compute_simd();
    break;

  // new functions
  case COMPUTE_B16:
    // printf("COMPUTE_B16\n");
    compute_b16();
    break;
  case COMPUTE_MKNMKN_B32_LU8:
    // printf("COMPUTE_MKNMKN_B32_LU8\n");
    compute_mknmkn_b32_lu8();
    break;
  case COMPUTE_MKNMKN_B32_Bj64_LU8:
    // printf("COMPUTE_MKNMKN_B32_Bj64_LU8\n");
    compute_mknmkn_b32_Bj64_lu8();
    break;
  case COMPUTE_MKNMKN_B32_Bj128_LU8:
    // printf("COMPUTE_MKNMKN_B32_Bj128_LU8\n");
    compute_mknmkn_b32_Bj128_lu8();
    break;
  default:
    printf("Unreachable!");
    return 0;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  return elapsed(start, end);
}
