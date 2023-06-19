/*
    Modified from https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL
*/

static const char * source = R""""(
static __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#if IS_FLOAT
#define WRITE_FUNC write_imagef
#define READ_FUNC read_imagef
#else
#define WRITE_FUNC write_imageui
#define READ_FUNC read_imageui
#endif

#if USE_OLD_PSCRN
static void elliott(FLOAT_TYPE8 * data, const int n) {
    for (int i = 0; i < n; i++)
        data[i] = native_divide(data[i], 1.f + fabs(data[i]));
}

static void dotProd(const FLOAT_TYPE8 * data, __constant WEIGHTS_FLOAT_TYPE * weights, FLOAT_TYPE8 * vals, const int n, const int len) {
    for (int i = 0; i < n; i++) {
        FLOAT_TYPE8 sum = 0.f;
        for (int j = 0; j < len; j++)
            sum += data[j] * weights[mad24(i, len, j)];

        vals[i] = sum + weights[mad24(n, len, i)];
    }
}

static FLOAT_TYPE8 prescreen(const __local FLOAT_TYPE (* input)[INPUT_WIDTH], int8 * flag, __constant WEIGHTS_FLOAT_TYPE * weights) {
    FLOAT_TYPE8 temp[12];

    for (int i = 0; i < 4; i++) {
        FLOAT_TYPE8 sum = 0.f;
        int j = 0;

        for (int y = 0; y < 4; y++) {
            FLOAT_TYPE8 pixel = vload8(0, input[y]);

            for (int x = 0; x < 12 - 1; x++) {
                sum += pixel * weights[mad24(i, 48, j++)];

                pixel = (FLOAT_TYPE8)(pixel.s1234, pixel.s567, input[y][8 + x]);
            }

            sum += pixel * weights[mad24(i, 48, j++)];
        }

        temp[i] = sum + weights[4 * 48 + i];
    }

    const FLOAT_TYPE8 t = temp[0];
    elliott(temp, 4);
    temp[0] = t;
    dotProd(temp, weights + 4 * 49, temp + 4, 4, 4);
    elliott(temp + 4, 4);
    dotProd(temp, weights + 4 * 49 + 4 * 5, temp + 8, 4, 8);

    *flag = (max(temp[10], temp[11]) <= max(temp[8], temp[9]));

    return 0.59375f * (vload8(0, input[1] + 5) + vload8(0, input[2] + 5)) - 0.09375f * (vload8(0, input[0] + 5) + vload8(0, input[3] + 5));
}
#endif

#if USE_NEW_PSCRN
static FLOAT_TYPE8 prescreen(const __local FLOAT_TYPE (* input)[INPUT_WIDTH], int8 * flag, __constant WEIGHTS_FLOAT_TYPE * weights) {
    __constant short * ws = (__constant short *)weights;
    __constant WEIGHTS_FLOAT_TYPE * wf = (__constant WEIGHTS_FLOAT_TYPE *)&ws[4 * 64];
    FLOAT_TYPE temp1[8], temp2[8];

    for (int i = 0; i < 4; i++) {
        FLOAT_TYPE sum1 = 0.f, sum2 = 0.f;
        int j = 0;

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 16; x++) {
                sum1 += input[y][x] * ws[(i << 3) + ((j >> 3) << 5) + (j & 7)];
                sum2 += input[y][4 + x] * ws[(i << 3) + ((j >> 3) << 5) + (j & 7)];
                j++;
            }
        }

        const FLOAT_TYPE t1 = sum1 * wf[i] + wf[4 + i];
        const FLOAT_TYPE t2 = sum2 * wf[i] + wf[4 + i];
        temp1[i] = native_divide(t1, 1.f + fabs(t1));
        temp2[i] = native_divide(t2, 1.f + fabs(t2));
    }

    for (int i = 0; i < 4; i++) {
        FLOAT_TYPE sum1 = 0.f, sum2 = 0.f;
        for (int j = 0; j < 4; j++) {
            sum1 += temp1[j] * wf[8 + i + (j << 2)];
            sum2 += temp2[j] * wf[8 + i + (j << 2)];
        }

        temp1[4 + i] = sum1 + wf[8 + 16 + i];
        temp2[4 + i] = sum2 + wf[8 + 16 + i];
    }

    for (int i = 0; i < 4; i++) {
        ((int *)flag)[i] = select(0, -1, temp1[4 + i] > 0.f);
        ((int *)flag)[4 + i] = select(0, -1, temp2[4 + i] > 0.f);
    }

    return 0.59375f * (vload8(0, input[1] + 6) + vload8(0, input[2] + 6)) - 0.09375f * (vload8(0, input[0] + 6) + vload8(0, input[3] + 6));
}
#endif

#if !(USE_OLD_PSCRN || USE_NEW_PSCRN)
static FLOAT_TYPE8 prescreen(const __local FLOAT_TYPE (* input)[INPUT_WIDTH], int8 * flag, __constant WEIGHTS_FLOAT_TYPE * weights) {
}
#endif

static FLOAT_TYPE8 predict(const __local FLOAT_TYPE (* input)[INPUT_WIDTH], __read_only image1d_buffer_t weights) {
    FLOAT_TYPE8 sum = 0.f, sumsq = 0.f;

    #pragma unroll
    for (int y = 0; y < YDIA; y++) {
        FLOAT_TYPE8 pixel = vload8(0, input[y]);

        #pragma unroll
        for (int x = 0; x < XDIA - 1; x++) {
            sum += pixel;
            sumsq += pixel * pixel;

            pixel = (FLOAT_TYPE8)(pixel.s1234, pixel.s567, input[y][8 + x]);
        }

        sum += pixel;
        sumsq += pixel * pixel;
    }

    const FLOAT_TYPE8 mstd0 = sum * SCALE_ASIZE;
    FLOAT_TYPE8 mstd1 = sumsq * SCALE_ASIZE - mstd0 * mstd0;
    const int8 cond = (mstd1 <= FLT_EPSILON);
    mstd1 = select(native_sqrt(mstd1), 0.f, cond);
    const FLOAT_TYPE8 mstd2 = select(native_recip(mstd1), 0.f, cond);

    FLOAT_TYPE8 mstd3 = 0.f;

    #pragma unroll 1
    for (int q = 0; q < QUAL; q++) {
        const int weightsOffset = mul24(DIMS1, q);
        FLOAT_TYPE8 vsum = 0.f, wsum = 0.f;

        #pragma unroll 1
        for (int i = 0; i < NNS; i++) {
            FLOAT_TYPE8 sum1 = 0.f, sum2 = 0.f;
            int j = 0;

            #pragma unroll 1
            for (int y = 0; y < YDIA; y++) {
                FLOAT_TYPE8 pixel = vload8(0, input[y]);

                #pragma unroll
                for (int x = 0; x < XDIA - 1; x++) {
                    sum1 += pixel * read_imagef(weights, weightsOffset + mad24(i, ASIZE, j)).x;
                    sum2 += pixel * read_imagef(weights, weightsOffset + mad24(NNS + i, ASIZE, j++)).x;

                    pixel = (FLOAT_TYPE8)(pixel.s1234, pixel.s567, input[y][8 + x]);
                }

                sum1 += pixel * read_imagef(weights, weightsOffset + mad24(i, ASIZE, j)).x;
                sum2 += pixel * read_imagef(weights, weightsOffset + mad24(NNS + i, ASIZE, j++)).x;
            }

            sum1 = native_exp(clamp(sum1 * mstd2 + read_imagef(weights, weightsOffset + NNS2 * ASIZE + i).x, -80.f, 80.f));
            sum2 = sum2 * mstd2 + read_imagef(weights, weightsOffset + NNS2 * ASIZE + NNS + i).x;

            vsum += sum1 * native_divide(sum2, 1.f + fabs(sum2));
            wsum += sum1;
        }

        mstd3 += select(mstd0, native_divide(5.f * vsum, wsum) * mstd1 + mstd0, wsum > 1e-10f);
    }

    return mstd3 * SCALE_QUAL;
}

__kernel __attribute__((reqd_work_group_size(4, 16, 1)))
void filter(__read_only image2d_t src, __write_only image2d_t dst, __constant WEIGHTS_FLOAT_TYPE * weights0, __read_only image1d_buffer_t weights1, const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight, const int field_n, const int off, const int swap, const int offY) {
    const int globalX = get_global_id(0);
    const int globalY = get_global_id(1);
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);

    const int _srcX = -XDIAD2M1 + 32 * (int)get_group_id(0) + localX;
    const int _srcY = field_n - Y_OFFSET + Y_STEP * globalY + offY;
    const int _dstX = 8 * globalX;
    const int dstYCopy = off + 2 * globalY;
    const int dstY = field_n + 2 * globalY;

    __local FLOAT_TYPE input[INPUT_HEIGHT][INPUT_WIDTH];

    for (int y = localY, j = 0; y < INPUT_HEIGHT; y += 16, j++) {
        int srcY = _srcY + Y_STRIDE * j;
        if (srcY < 0)
            srcY = abs(srcY) + Y_STEP * off;
        else if (srcY >= srcHeight)
            srcY = 2 * srcHeight - srcY - 2 * Y_STEP;

        for (int x = localX, i = 0; x < INPUT_WIDTH; x += 4, i++) {
            int srcX = abs(_srcX + 4 * i);
            if (srcX >= srcWidth)
                srcX = 2 * srcWidth - srcX - 2;

            input[y][x] = READ_FUNC(src, sampler, select((int2)(srcX, srcY), (int2)(srcY, srcX), (int2)swap)).x;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    FLOAT_TYPE8 output;
#if USE_OLD_PSCRN || USE_NEW_PSCRN
    int8 flag;
    output = prescreen((const __local FLOAT_TYPE (*)[INPUT_WIDTH])&input[YDIAD2M1 - 1 + localY][XDIAD2M1 - PSCRN_OFFSET + 8 * localX], &flag, weights0);
    if (!all(flag))
#endif
        output = predict((const __local FLOAT_TYPE (*)[INPUT_WIDTH])&input[localY][X_OFFSET + 8 * localX], weights1);

    if (_dstX >= 0 && dstY >= 0 && dstY < dstHeight) {
        for (int i = 0; i < 8; i++) {
            const int dstX = _dstX + i;
            if (dstX < dstWidth) {
                WRITE_FUNC(dst, select((int2)(dstX, dstYCopy), (int2)(dstYCopy, dstX), (int2)swap), input[YDIAD2M1 + localY + off][XDIAD2M1 + 8 * localX + i]);

                #if IS_FLOAT
                    WRITE_FUNC(dst, select((int2)(dstX, dstY), (int2)(dstY, dstX), (int2)swap), ((const FLOAT_TYPE *)&output)[i]);
                #else
                    WRITE_FUNC(dst, select((int2)(dstX, dstY), (int2)(dstY, dstX), (int2)swap), clamp((int)(((const FLOAT_TYPE *)&output)[i] + 0.5f), 0, PEAK));
                #endif
            }
        }
    }
}
)"""";