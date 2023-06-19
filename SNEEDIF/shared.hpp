#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "half_float/umHalf.h"
#include "vectorclass/vectorclass.h"

#include <VSHelper4.h>
#include <VapourSynth4.h>
#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_USE_OFFLINE_CACHE
#include <boost/compute/core.hpp>
#include <boost/compute/utility/dim.hpp>
namespace compute = boost::compute;

struct NNEDI3Data {
        VSNode *node;
        VSVideoInfo vi;
        int field;
        bool process[3];
        int dims1, dims0, dims0new;
        size_t **globalWorkSize;
        compute::device device;
        compute::context context;
        compute::program program;
        std::unordered_map<std::thread::id, compute::command_queue> queue;
        std::unordered_map<std::thread::id, compute::kernel> kernel;
        std::unordered_map<std::thread::id, compute::image2d> src, dst, tmp;
        compute::buffer weights0, weights1Buffer;
        cl_mem weights1;
};

void VS_CC nnedi3Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi);

static inline int roundds(const double f) noexcept {
    return (f - std::floor(f) >= 0.5) ? std::min(static_cast<int>(std::ceil(f)), 32767)
                                      : std::max(static_cast<int>(std::floor(f)), -32768);
}