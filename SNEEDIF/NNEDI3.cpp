/*
    Modified from https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL
*/

#include "NNEDI3.cl"

#include "shared.hpp"

static constexpr int numNSIZE = 7;
static constexpr int numNNS = 5;
static constexpr int xdiaTable[numNSIZE] = { 8, 16, 32, 48, 8, 16, 32 };
static constexpr int ydiaTable[numNSIZE] = { 6, 6, 6, 6, 4, 4, 4 };
static constexpr int nnsTable[numNNS] = { 16, 32, 64, 128, 256 };

template<typename T, bool dw, bool dh, bool transpose_first>
static void filter(
    const VSFrame *src, VSFrame *dst, const int field_n, const NNEDI3Data * const VS_RESTRICT d, const VSAPI *vsapi
) {
    auto threadId = std::this_thread::get_id();

    auto queue = d->queue.at(threadId);
    auto kernel = d->kernel.at(threadId);
    auto srcImage = d->src.at(threadId);
    auto dstImage = d->dst.at(threadId);
    auto tmpImage = d->tmp.at(threadId);

    for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int srcWidth = vsapi->getFrameWidth(src, plane);
            const int srcHeight = vsapi->getFrameHeight(src, plane);
            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);

            const T *srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
            T *VS_RESTRICT dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

            constexpr size_t localWorkSize[] = { 4, 16 };
            const int shiftY = dh ? (plane && d->vi.format.subSamplingH ? (1 << d->vi.format.subSamplingH) : 4) : 0;
            const int shiftX = dw ? (plane && d->vi.format.subSamplingW ? (1 << d->vi.format.subSamplingW) : 4) : 0;


            queue.enqueue_write_image(
                srcImage, compute::dim(0, 0), compute::dim(srcWidth, srcHeight), srcp, vsapi->getStride(src, plane)
            );

            if constexpr (dh && dw) {
                if constexpr (transpose_first) {
                    kernel.set_args(
                        srcImage, tmpImage, d->weights0, d->weights1, srcHeight, srcWidth, srcHeight, dstWidth, field_n,
                        1 - field_n, -1, shiftX
                    );
                } else {
                    kernel.set_args(
                        srcImage, tmpImage, d->weights0, d->weights1, srcWidth, srcHeight, srcWidth, dstHeight, field_n,
                        1 - field_n, 0, shiftY
                    );
                }

                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, d->globalWorkSize[plane], localWorkSize);

                if constexpr (transpose_first) {
                    kernel.set_args(
                        tmpImage, dstImage, d->weights0, d->weights1, dstWidth, srcHeight, dstWidth, dstHeight, field_n,
                        1 - field_n, 0, shiftY
                    );
                } else {
                    kernel.set_args(
                        tmpImage, dstImage, d->weights0, d->weights1, dstHeight, srcWidth, dstHeight, dstWidth, field_n,
                        1 - field_n, -1, shiftX
                    );
                }

                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, d->globalWorkSize[plane + 3], localWorkSize);
            } else {
                if constexpr (dw) {
                    kernel.set_args(
                        srcImage, dstImage, d->weights0, d->weights1, srcHeight, srcWidth, dstHeight, dstWidth, field_n,
                        1 - field_n, -1, shiftX
                    );
                } else {
                    kernel.set_args(
                        srcImage, dstImage, d->weights0, d->weights1, srcWidth, srcHeight, dstWidth, dstHeight, field_n,
                        1 - field_n, 0, shiftY
                    );
                }

                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, d->globalWorkSize[plane], localWorkSize);
            }

            queue.enqueue_read_image(
                dstImage, compute::dim(0, 0), compute::dim(dstWidth, dstHeight), dstp, vsapi->getStride(dst, plane)
            );
        }
    }
}

template<bool dw, bool dh, bool transpose_first>
static const VSFrame *VS_CC nnedi3GetFrame(
    int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi
) {
    NNEDI3Data *d = static_cast<NNEDI3Data *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->prop_node, frameCtx);
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto threadId = std::this_thread::get_id();

        if (!d->queue.count(threadId)) {
            try {

                d->queue.emplace(threadId, compute::command_queue { d->context, d->device });

                cl_image_format imageFormat { CL_R, CL_SIGNED_INT8 };

                if (d->vi.format.sampleType == stInteger) {
                    if (d->vi.format.bytesPerSample == 1)
                        imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
                    else if (d->vi.format.bytesPerSample == 2)
                        imageFormat.image_channel_data_type = CL_UNSIGNED_INT16;
                    else if (d->vi.format.bytesPerSample == 4)
                        imageFormat.image_channel_data_type = CL_UNSIGNED_INT32;
                } else {
                    if (d->vi.format.bytesPerSample == 2)
                        imageFormat.image_channel_data_type = CL_HALF_FLOAT;
                    else if (d->vi.format.bytesPerSample == 4)
                        imageFormat.image_channel_data_type = CL_FLOAT;
                }

                d->kernel.emplace(threadId, d->program.create_kernel("filter"));

                d->src.emplace(
                    threadId,
                    compute::image2d { d->context, static_cast<size_t>(dw ? d->vi.width / 2 + 8 : d->vi.width),
                                       static_cast<size_t>(dh ? d->vi.height / 2 + 8 : d->vi.height),
                                       compute::image_format { imageFormat },
                                       CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY }
                );

                d->dst.emplace(
                    threadId,
                    compute::image2d { d->context, static_cast<size_t>(d->vi.width), static_cast<size_t>(d->vi.height),
                                       compute::image_format { imageFormat },
                                       CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY }
                );

                d->tmp.emplace(
                    threadId, (dh && dw) ? compute::image2d { d->context,
                                                              static_cast<size_t>(std::max(d->vi.width, d->vi.height)),
                                                              static_cast<size_t>(std::max(d->vi.width, d->vi.height)),
                                                              compute::image_format { imageFormat },
                                                              CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS }
                                         : compute::image2d {}
                );
            } catch (const std::string &error) {
                vsapi->setFilterError(("NNEDI3: " + error).c_str(), frameCtx);
                return nullptr;
            } catch (const compute::opencl_error &error) {
                vsapi->setFilterError(("NNEDI3: " + error.error_string()).c_str(), frameCtx);
                return nullptr;
            }
        }

        const VSFrame *prop_src = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->prop_node, frameCtx);
        const VSFrame *src = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);

        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, prop_src, core);

        int field = d->field;
        if (field > 1)
            field -= 2;

        int err;
        const int fieldBased =
            vsh::int64ToIntS(vsapi->mapGetInt(vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err));
        if (fieldBased == 1)
            field = 0;
        else if (fieldBased == 2)
            field = 1;

        int field_n;
        if (d->field > 1) {
            if (n & 1)
                field_n = (field == 0);
            else
                field_n = (field == 1);
        } else {
            field_n = field;
        }

        try {
            if (d->vi.format.sampleType == stFloat) {
                if (d->vi.format.bytesPerSample == 2)
                    filter<half, dw, dh, transpose_first>(src, dst, field_n, d, vsapi);
                else if (d->vi.format.bytesPerSample == 4)
                    filter<float, dw, dh, transpose_first>(src, dst, field_n, d, vsapi);
            } else {
                if (d->vi.format.bytesPerSample == 1)
                    filter<uint8_t, dw, dh, transpose_first>(src, dst, field_n, d, vsapi);
                else if (d->vi.format.bytesPerSample == 2)
                    filter<uint16_t, dw, dh, transpose_first>(src, dst, field_n, d, vsapi);
                else if (d->vi.format.bytesPerSample == 2)
                    filter<uint32_t, dw, dh, transpose_first>(src, dst, field_n, d, vsapi);
            }
        } catch (const compute::opencl_error &error) {
            vsapi->setFilterError(("NNEDI3: " + error.error_string()).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        VSMap *props = vsapi->getFramePropertiesRW(dst);

        if (d->field > 1) {
            int errNum, errDen;
            int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
            int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
            if (!errNum && !errDen) {
                vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
            }
        }

        vsapi->mapSetInt(props, "_FieldBased", 0, maReplace);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC nnedi3Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    NNEDI3Data *d = static_cast<NNEDI3Data *>(instanceData);

    vsapi->freeNode(d->node);

    clReleaseMemObject(d->weights1);

    d->queue.clear();
    d->kernel.clear();
    d->src.clear();
    d->dst.clear();
    d->tmp.clear();

    clReleaseDevice(d->device.id());

    delete[] d->globalWorkSize;

    delete d;
}

void VS_CC nnedi3Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<NNEDI3Data> d = std::make_unique<NNEDI3Data>();
    int err;
    VSFilterGetFrame getFrame;

    d->node = d->prop_node = vsapi->mapGetNode(in, "clip", 0, nullptr);

    d->vi = *vsapi->getVideoInfo(d->prop_node);

    try {
        if (!vsh::isConstantVideoFormat(&d->vi))
            throw std::string { "only constant format input supported" };

        d->field = vsh::int64ToIntS(vsapi->mapGetInt(in, "field", 0, nullptr));

        bool dh = !!vsapi->mapGetInt(in, "dh", 0, &err);
        bool dw = !!vsapi->mapGetInt(in, "dw", 0, &err);
        bool transpose_first = !!vsapi->mapGetInt(in, "transpose_first", 0, &err);

        if (dw || dh) {
            VSMap *args = vsapi->createMap();
            vsapi->mapSetNode(args, "clip", d->prop_node, maReplace);
            vsapi->mapSetInt(args, "width", d->vi.width + (8 * dw), maReplace);
            vsapi->mapSetInt(args, "height", d->vi.height + (8 * dh), maReplace);
            vsapi->mapSetFloat(args, "src_width", d->vi.width + (8 * dw), maReplace);
            vsapi->mapSetFloat(args, "src_height", d->vi.height + (8 * dh), maReplace);
            vsapi->mapSetFloat(args, "src_left", -4 * dw, maReplace);
            vsapi->mapSetFloat(args, "src_top", -4 * dh, maReplace);

            VSMap *ret = vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }

        const int m = vsapi->mapNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = vsh::int64ToIntS(vsapi->mapGetInt(in, "planes", i, nullptr));

            if (n < 0 || n >= d->vi.format.numPlanes)
                throw std::string { "plane index out of range" };

            if (d->process[n])
                throw std::string { "plane specified twice" };

            d->process[n] = true;
        }

        int nsize = vsh::int64ToIntS(vsapi->mapGetInt(in, "nsize", 0, &err));
        if (err)
            nsize = 6;

        int nns = vsh::int64ToIntS(vsapi->mapGetInt(in, "nns", 0, &err));
        if (err)
            nns = 1;

        int qual = vsh::int64ToIntS(vsapi->mapGetInt(in, "qual", 0, &err));
        if (err)
            qual = 1;

        const int etype = vsh::int64ToIntS(vsapi->mapGetInt(in, "etype", 0, &err));

        int pscrn = vsh::int64ToIntS(vsapi->mapGetInt(in, "pscrn", 0, &err));
        if (err)
            pscrn = 1;

        bool float16_data = false;
        bool float16_weights = false;

        // bool float16_data = !!vsh::int64ToIntS(vsapi->mapGetInt(in, "float16_data", 0, &err));
        // if (err)
        float16_data = d->vi.format.sampleType == stFloat && d->vi.format.bitsPerSample == 2;

        // bool float16_weights = !!vsh::int64ToIntS(vsapi->mapGetInt(in, "float16_weights", 0, &err));
        // if (err)
        //     float16_weights = true;

        int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device", 0, &err));
        if (err)
            device_id = -1;

        if (d->field < 0 || d->field > 3)
            throw std::string { "field must be 0, 1, 2 or 3" };

        if (!dh && (d->vi.height & 1))
            throw std::string { "height must be mod 2 when dh=False" };

        if (dh && d->field > 1)
            throw std::string { "field must be 0 or 1 when dh=True" };

        if (dw && d->field > 1)
            throw std::string { "field must be 0 or 1 when dw=True" };

        if (nsize < 0 || nsize > 6)
            throw std::string { "nsize must be 0, 1, 2, 3, 4, 5 or 6" };

        if (nns < 0 || nns > 4)
            throw std::string { "nns must be 0, 1, 2, 3 or 4" };

        if (qual < 1 || qual > 2)
            throw std::string { "qual must be 1 or 2" };

        if (etype < 0 || etype > 1)
            throw std::string { "etype must be 0 or 1" };

        if (d->vi.format.sampleType == stInteger) {
            if (pscrn < 0 || pscrn > 2)
                throw std::string { "pscrn must be 0, 1 or 2" };
        } else {
            if (pscrn < 0 || pscrn > 1)
                throw std::string { "pscrn must be 0 or 1 for float input" };
        }

        if (device_id >= static_cast<int>(compute::system::device_count()))
            throw std::string { "device index out of range" };

        d->device = compute::system::default_device();
        if (device_id > -1)
            d->device = compute::system::devices().at(device_id);
        d->context = compute::context { d->device };

        VSCoreInfo info;
        vsapi->getCoreInfo(core, &info);

        d->queue.reserve(info.numThreads);
        d->kernel.reserve(info.numThreads);
        d->src.reserve(info.numThreads);
        d->dst.reserve(info.numThreads);
        d->tmp.reserve(info.numThreads);

        if (d->field > 1) {
            if (d->vi.numFrames > INT_MAX / 2)
                throw std::string { "resulting clip is too long" };
            d->vi.numFrames *= 2;

            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }

        if (dh)
            d->vi.height *= 2;

        if (dw)
            d->vi.width *= 2;

        const int peak = (1 << d->vi.format.bitsPerSample) - 1;

        const std::string pluginPath { vsapi->getPluginPath(static_cast<VSPlugin *>(userData)) };
        std::string weightsPath { pluginPath.substr(0, pluginPath.find_last_of('/')) + "/nnedi3_weights.bin" };

        FILE *weightsFile = nullptr;
#ifdef _WIN32
        const int requiredSize = MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, nullptr, 0);
        std::unique_ptr<wchar_t[]> wbuffer = std::make_unique<wchar_t[]>(requiredSize);
        MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, wbuffer.get(), requiredSize);
        weightsFile = _wfopen(wbuffer.get(), L"rb");
#else
        weightsFile = std::fopen(weightsPath.c_str(), "rb");
#endif

#if !defined(_WIN32) && defined(NNEDI3_DATADIR)
        if (!weightsFile) {
            weightsPath = std::string { NNEDI3_DATADIR } + "/nnedi3_weights.bin";
            weightsFile = std::fopen(weightsPath.c_str(), "rb");
        }
#endif
        if (!weightsFile)
            throw std::string { "error opening file " + weightsPath + " (" + std::strerror(errno) + ")" };

        if (std::fseek(weightsFile, 0, SEEK_END)) {
            std::fclose(weightsFile);
            throw std::string { "error seeking to the end of file " + weightsPath + " (" + std::strerror(errno) + ")" };
        }

        constexpr long correctSize = 13574928;  // Version 0.9.4 of the Avisynth plugin
        const long weightsSize = std::ftell(weightsFile);

        if (weightsSize == -1) {
            std::fclose(weightsFile);
            throw std::string { "error determining the size of file " + weightsPath + " (" + std::strerror(errno) +
                                ")" };
        } else if (weightsSize != correctSize) {
            std::fclose(weightsFile);
            throw std::string { "incorrect size of file " + weightsPath + ". Should be " + std::to_string(correctSize) +
                                " bytes, but got " + std::to_string(weightsSize) + " bytes instead" };
        }

        std::rewind(weightsFile);

        float *bdata = reinterpret_cast<float *>(malloc(correctSize));
        const size_t bytesRead = std::fread(bdata, 1, correctSize, weightsFile);

        if (bytesRead != correctSize) {
            std::fclose(weightsFile);
            free(bdata);
            throw std::string { "error reading file " + weightsPath + ". Should read " + std::to_string(correctSize) +
                                " bytes, but read " + std::to_string(bytesRead) + " bytes instead" };
        }

        std::fclose(weightsFile);

        constexpr int dims0 = 49 * 4 + 5 * 4 + 9 * 4;
        constexpr int dims0new = 4 * 65 + 4 * 5;
        const int dims1 = nnsTable[nns] * 2 * (xdiaTable[nsize] * ydiaTable[nsize] + 1);
        int dims1tsize = 0, dims1offset = 0;

        for (int j = 0; j < numNNS; j++) {
            for (int i = 0; i < numNSIZE; i++) {
                if (i == nsize && j == nns)
                    dims1offset = dims1tsize;
                dims1tsize += nnsTable[j] * 2 * (xdiaTable[i] * ydiaTable[i] + 1) * 2;
            }
        }

        float *weights0_b = new float[std::max(dims0, dims0new)];
        float *weights1_b = new float[dims1 * 2];

        // Adjust prescreener weights
        if (pscrn == 2) {  // using new prescreener
            int *offt = reinterpret_cast<int *>(calloc(4 * 64, sizeof(int)));
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 64; k++)
                    offt[j * 64 + k] = ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
            }

            const float *bdw = bdata + dims0 + dims0new * (pscrn - 2);
            short *ws = reinterpret_cast<short *>(weights0_b);
            float *wf = reinterpret_cast<float *>(&ws[4 * 64]);
            double mean[4] = { 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j = 0; j < 4; j++) {
                double cmean = 0.0;
                for (int k = 0; k < 64; k++)
                    cmean += bdw[offt[j * 64 + k]];

                mean[j] = cmean / 64.0;
            }

            const double half = peak / 2.0;

            // Factor mean removal and 1.0/half scaling into first layer weights. scale to int16 range
            for (int j = 0; j < 4; j++) {
                double mval = 0.0;
                for (int k = 0; k < 64; k++)
                    mval = std::max(mval, std::abs((bdw[offt[j * 64 + k]] - mean[j]) / half));

                const double scale = 32767.0 / mval;
                for (int k = 0; k < 64; k++)
                    ws[offt[j * 64 + k]] = roundds(((bdw[offt[j * 64 + k]] - mean[j]) / half) * scale);

                wf[j] = static_cast<float>(mval / 32767.0);
            }

            memcpy(wf + 4, bdw + 4 * 64, (dims0new - 4 * 64) * sizeof(float));
            free(offt);
        } else if (pscrn == 1) {  // using old prescreener
            double mean[4] = { 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j = 0; j < 4; j++) {
                double cmean = 0.0;
                for (int k = 0; k < 48; k++)
                    cmean += bdata[j * 48 + k];

                mean[j] += cmean / 48.0;
            }

            const double half = (d->vi.format.sampleType == stInteger ? peak : 1.0) / 2.0;

            // Factor mean removal and 1.0/half scaling into first layer weights
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 48; k++)
                    weights0_b[j * 48 + k] = static_cast<float>((bdata[j * 48 + k] - mean[j]) / half);
            }

            memcpy(weights0_b + 4 * 48, bdata + 4 * 48, (dims0 - 4 * 48) * sizeof(float));
        }

        // Adjust prediction weights
        for (int i = 0; i < 2; i++) {
            const float *bdataT = bdata + dims0 + dims0new * 3 + dims1tsize * etype + dims1offset + i * dims1;
            float *weightsT = weights1_b + i * dims1;
            const int nnst = nnsTable[nns];
            const int asize = xdiaTable[nsize] * ydiaTable[nsize];
            const int boff = nnst * 2 * asize;
            double *mean = reinterpret_cast<double *>(calloc(asize + 1 + nnst * 2, sizeof(double)));

            // Calculate mean weight of each neuron (ignore bias)
            for (int j = 0; j < nnst * 2; j++) {
                double cmean = 0.0;
                for (int k = 0; k < asize; k++)
                    cmean += bdataT[j * asize + k];

                mean[asize + 1 + j] = cmean / asize;
            }

            // Calculate mean softmax neuron
            for (int j = 0; j < nnst; j++) {
                for (int k = 0; k < asize; k++)
                    mean[k] += bdataT[j * asize + k] - mean[asize + 1 + j];
                mean[asize] += bdataT[boff + j];
            }
            for (int j = 0; j < asize + 1; j++)
                mean[j] /= nnst;

            // Factor mean removal into weights, and remove global offset from softmax neurons
            for (int j = 0; j < nnst * 2; j++) {
                for (int k = 0; k < asize; k++) {
                    const double q = (j < nnst) ? mean[k] : 0.0;
                    weightsT[j * asize + k] = static_cast<float>(bdataT[j * asize + k] - mean[asize + 1 + j] - q);
                }
                weightsT[boff + j] = static_cast<float>(bdataT[boff + j] - (j < nnst ? mean[asize] : 0.0));
            }

            free(mean);
        }

        free(bdata);

        const int xdia = xdiaTable[nsize];
        const int ydia = ydiaTable[nsize];
        const int asize = xdiaTable[nsize] * ydiaTable[nsize];
        const int xdiad2m1 = std::max(xdia, (pscrn == 1) ? 12 : 16) / 2 - 1;
        const int ydiad2m1 = ydia / 2 - 1;
        const int xOffset = (xdia == 8) ? (pscrn == 1 ? 2 : 4) : 0;
        const int inputWidth = std::max(xdia, (pscrn == 1) ? 12 : 16) + 32 - 1;
        const int inputHeight = ydia + 16 - 1;
        const float scaleAsize = 1.0f / asize;
        const float scaleQual = 1.0f / qual;

        d->dims0 = dims0;
        d->dims1 = dims1;
        d->dims0new = dims0new;

        int buffer_rw_type = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
        int buff0_size = std::max(d->dims0, d->dims0new);
        int dims2 = d->dims1 * 2;

        if (float16_weights) {
            half *weights0_bh = new half[buff0_size];
            half *weights1_bh = new half[dims2];

            for (int i = 0; i < buff0_size; i++)
                weights0_bh[i] = weights0_b[i];

            for (int i = 0; i < dims2; i++)
                weights1_bh[i] = weights1_b[i];

            d->weights0 = compute::buffer { d->context, buff0_size * sizeof(cl_half), buffer_rw_type, weights0_bh };
            d->weights1Buffer = compute::buffer { d->context, dims2 * sizeof(cl_half), buffer_rw_type, weights1_bh };
        } else {
            d->weights0 = compute::buffer { d->context, buff0_size * sizeof(cl_float), buffer_rw_type, weights0_b };
            d->weights1Buffer = compute::buffer { d->context, dims2 * sizeof(cl_float), buffer_rw_type, weights1_b };
        }

        {
            cl_image_format format = { CL_R, CL_FLOAT };
            if (float16_weights)
                format.image_channel_data_type = CL_HALF_FLOAT;

            cl_image_desc desc;
            desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width = d->dims1 * 2;
            desc.image_height = 1;
            desc.image_depth = 1;
            desc.image_array_size = 0;
            desc.image_row_pitch = 0;
            desc.image_slice_pitch = 0;
            desc.num_mip_levels = 0;
            desc.num_samples = 0;
#ifdef BOOST_COMPUTE_CL_VERSION_2_0
            desc.mem_object = d->weights1Buffer.get();
#else
            desc.buffer = d->weights1Buffer.get();
#endif

            cl_int error = 0;

            d->weights1 = clCreateImage(d->context, 0, &format, &desc, nullptr, &error);
            if (!d->weights1)
                BOOST_THROW_EXCEPTION(compute::opencl_error(error));
        }

        const auto get_sizes = [](size_t x, size_t y) {
            return std::make_pair<size_t, size_t>(
                static_cast<size_t>((x / 8 + 3) & -4), static_cast<size_t>((y + 15) & -16)
            );
        };

        d->globalWorkSize = (size_t **) new size_t[6];

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            int dstWidth = d->vi.width >> (plane ? d->vi.format.subSamplingW : 0);
            int dstHeight = d->vi.height >> (plane ? d->vi.format.subSamplingH : 0);

            for (int i = 0; i < 2; i++) {
                std::pair<size_t, size_t> sizes;

                if (dw && dh) {
                    sizes = i == 0
                              ? (transpose_first ? get_sizes(dstHeight / 2 + 8, dstWidth / 2)
                                                 : get_sizes(dstWidth / 2 + 8, dstHeight / 2))
                              : (transpose_first ? get_sizes(dstWidth, dstHeight / 2) : get_sizes(dstHeight, dstWidth));
                } else {
                    if (i != 0)
                        continue;

                    sizes = dw ? get_sizes(dstHeight, dstWidth / 2) : get_sizes(dstWidth, dstHeight / 2);
                }

                d->globalWorkSize[plane + 3 * i] = new size_t[2];
                d->globalWorkSize[plane + 3 * i][0] = sizes.first;
                d->globalWorkSize[plane + 3 * i][1] = sizes.second;
            }
        }

        if (static_cast<size_t>(dims1 * 2) > d->device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE))
            throw std::string {
                "the device's image max buffer size is too small. Reduce nsize/nns...or buy a new graphics card"
            };

        try {
            std::ostringstream options;
            options.imbue(std::locale { "C" });
            options.precision(16);
            options.setf(std::ios::fixed, std::ios::floatfield);
            options << "-cl-denorms-are-zero -cl-fast-relaxed-math -Werror";
            options << " -D QUAL=" << qual;
            options << " -D USE_OLD_PSCRN=" << (pscrn == 1 ? "1" : "0");
            options << " -D USE_NEW_PSCRN=" << (pscrn == 2 ? "1" : "0");
            options << " -D PSCRN_OFFSET=" << (pscrn == 1 ? 5 : 6);
            options << " -D DIMS1=" << dims1;
            options << " -D NNS=" << nnsTable[nns];
            options << " -D NNS2=" << (nnsTable[nns] * 2);
            options << " -D XDIA=" << xdia;
            options << " -D YDIA=" << ydia;
            options << " -D ASIZE=" << asize;
            options << " -D XDIAD2M1=" << xdiad2m1;
            options << " -D YDIAD2M1=" << ydiad2m1;
            options << " -D X_OFFSET=" << xOffset;
            options << " -D INPUT_WIDTH=" << inputWidth;
            options << " -D INPUT_HEIGHT=" << inputHeight;
            options << " -D SCALE_ASIZE=" << scaleAsize << "f";
            options << " -D SCALE_QUAL=" << scaleQual << "f";
            options << " -D PEAK=" << peak;
            options << " -D IS_FLOAT=" << (d->vi.format.sampleType == stFloat ? "1" : "0");
            options << " -D FLOAT_TYPE=" << (float16_data ? "half" : "float");
            options << " -D FLOAT_TYPE8=" << (float16_data ? "half8" : "float8");
            options << " -D WEIGHTS_FLOAT_TYPE=" << (float16_weights ? "half" : "float");
            if (!(dh || dw)) {
                options << " -D Y_OFFSET=" << (ydia - 1);
                options << " -D Y_STEP=2";
                options << " -D Y_STRIDE=32";
            } else {
                options << " -D Y_OFFSET=" << (ydia / 2);
                options << " -D Y_STEP=1";
                options << " -D Y_STRIDE=16";
            }
            d->program = compute::program::build_with_source(source, d->context, options.str());
        } catch (const compute::opencl_error &error) {
            throw error.error_string() + "\n" + d->program.build_log();
        }

        if (dw && dh) {
            getFrame = transpose_first ? nnedi3GetFrame<true, true, true> : nnedi3GetFrame<true, true, false>;
        } else if (dw || dh) {
            getFrame = dw ? nnedi3GetFrame<true, false, false> : nnedi3GetFrame<false, true, false>;
        } else {
            getFrame = nnedi3GetFrame<false, false, false>;
        }
    } catch (const std::string &error) {
        vsapi->mapSetError(out, ("NNEDI3: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    } catch (const compute::no_device_found &error) {
        vsapi->mapSetError(out, (std::string { "NNEDI3: " } + error.what()).c_str());
        vsapi->freeNode(d->node);
        return;
    } catch (const compute::opencl_error &error) {
        vsapi->mapSetError(out, ("NNEDI3: " + error.error_string()).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[] = {
        {d->node, rpGeneral},
        {d->prop_node, rpGeneral},
    };
    vsapi->createVideoFilter(out, "NNEDI3", &d->vi, getFrame, nnedi3Free, fmParallel, deps, 2, d.get(), core);
    d.release();
}
