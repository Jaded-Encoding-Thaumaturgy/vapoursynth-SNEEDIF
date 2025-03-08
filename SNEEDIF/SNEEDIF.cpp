#include "shared.hpp"

void add_member_str(const VSAPI *vsapi, VSMap *map, const char *key, std::string str) {
    vsapi->mapSetData(map, key, str.c_str(), str.size(), dtUtf8, maReplace);
};

void add_member_int(const VSAPI *vsapi, VSMap *map, const char *key, int64_t value) {
    vsapi->mapSetInt(map, key, value, maReplace);
};

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "dev.setsugen.sneedif", "sneedif", "Setsugen No Ensemble of Edge Directed Interpolation Functions",
        VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin
    );

    vspapi->registerFunction(
        "NNEDI3",
        "clip:vnode;"
        "field:int;"
        "dh:int:opt;"
        "dw:int:opt;"
        "planes:int[]:opt;"
        "nsize:int:opt;"
        "nns:int:opt;"
        "qual:int:opt;"
        "etype:int:opt;"
        "pscrn:int:opt;"
        "transpose_first:int:opt;"
        // "float16_data:int:opt;"
        // "float16_weights:int:opt;"
        "device:int:opt;",
        "clip:vnode;", nnedi3Create, (void *) plugin, plugin
    );

    vspapi->registerFunction(
        "ListDevices", "", "numDevices:int;deviceNames:data[];platformNames:data[]",
        [](const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
            const auto devices = compute::system::devices();

            vsapi->mapSetInt(out, "numDevices", devices.size(), maReplace);

            for (auto &device : devices) {
                std::string name { device.name() };
                std::string platform { device.platform().name() };

                vsapi->mapSetData(out, "deviceNames", name.c_str(), name.size(), dtUtf8, maAppend);
                vsapi->mapSetData(out, "platformNames", platform.c_str(), platform.size(), dtUtf8, maAppend);
            }
        },
        NULL, plugin
    );

    vspapi->registerFunction(
        "PlatformInfo", "device:int:opt", "profile:data;version:data;name:data;vendor:data;",
        [](const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
            int err;

            int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device", 0, &err));
            if (err)
                device_id = -1;

            if (device_id >= static_cast<int>(compute::system::device_count()))
                throw std::string { "device index out of range" };

            compute::device device =
                device_id > -1 ? compute::system::devices().at(device_id) : compute::system::default_device();

            const auto platform = device.platform();

            add_member_str(vsapi, out, "profile", platform.get_info<CL_PLATFORM_PROFILE>());
            add_member_str(vsapi, out, "version", platform.get_info<CL_PLATFORM_VERSION>());
            add_member_str(vsapi, out, "name", platform.get_info<CL_PLATFORM_NAME>());
            add_member_str(vsapi, out, "vendor", platform.get_info<CL_PLATFORM_VENDOR>());
        },
        NULL, plugin
    );

    vspapi->registerFunction(
        "DeviceInfo", "device:int:opt",
        "name:data;"
        "vendor:data;"
        "profile:data;"
        "version:data;"
        "max_compute_units:int;"
        "max_work_group_size:int;"
        "max_work_item_sizes:int[];"
        "image2D_max_width:int;"
        "image2D_max_height:int;"
        "image_support:int;"
        "global_memory_cache_type:data;"
        "global_memory_cache:int;"
        "global_memory_size:int;"
        "max_constant_buffer_size:int;"
        "max_constant_arguments:int;"
        "local_memory_type:data;"
        "local_memory_size:int;"
        "available:int;"
        "compiler_available:int;"
        "linker_available:int;"
        "opencl_c_version:data;"
        "image_max_buffer_size:int;",
        [](const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
            int err;

            int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device", 0, &err));
            if (err)
                device_id = -1;

            if (device_id >= static_cast<int>(compute::system::device_count()))
                throw std::string { "device index out of range" };

            compute::device device =
                device_id > -1 ? compute::system::devices().at(device_id) : compute::system::default_device();

            const auto max_work_item_sizes = device.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            const auto global_mem_cache_type = device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();

            add_member_str(vsapi, out, "name", device.get_info<CL_DEVICE_NAME>());
            add_member_str(vsapi, out, "vendor", device.get_info<CL_DEVICE_VENDOR>());
            add_member_str(vsapi, out, "profile", device.get_info<CL_DEVICE_PROFILE>());
            add_member_str(vsapi, out, "version", device.get_info<CL_DEVICE_VERSION>());

            add_member_int(vsapi, out, "max_compute_units", device.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>());
            add_member_int(vsapi, out, "max_work_group_size", device.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

            for (int i = 0, l = max_work_item_sizes.size(); i < l; i++) {
                vsapi->mapSetInt(out, "max_work_item_sizes", max_work_item_sizes[i], maAppend);
            }

            add_member_int(vsapi, out, "image2D_max_width", device.get_info<CL_DEVICE_IMAGE2D_MAX_WIDTH>());
            add_member_int(vsapi, out, "image2D_max_height", device.get_info<CL_DEVICE_IMAGE2D_MAX_HEIGHT>());

            add_member_int(vsapi, out, "image_support", device.get_info<CL_DEVICE_IMAGE_SUPPORT>() ? 1 : 0);

            if (global_mem_cache_type == CL_NONE)
                add_member_str(vsapi, out, "global_memory_cache_type", "NONE");
            else if (global_mem_cache_type == CL_READ_ONLY_CACHE)
                add_member_str(vsapi, out, "global_memory_cache_type", "READ_ONLY_CACHE");
            else if (global_mem_cache_type == CL_READ_WRITE_CACHE)
                add_member_str(vsapi, out, "global_memory_cache_type", "READ_WRITE_CACHE");

            add_member_int(vsapi, out, "global_memory_cache size", device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>());
            add_member_int(vsapi, out, "global_memory_size", device.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>());

            add_member_int(
                vsapi, out, "max_constant_buffer_size", device.get_info<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()
            );
            add_member_int(vsapi, out, "max_constant_arguments", device.get_info<CL_DEVICE_MAX_CONSTANT_ARGS>());

            add_member_str(
                vsapi, out, "local_memory_type",
                device.get_info<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL"
            );
            add_member_int(vsapi, out, "local_memory_size", device.get_info<CL_DEVICE_LOCAL_MEM_SIZE>());

            add_member_int(vsapi, out, "available", device.get_info<CL_DEVICE_AVAILABLE>() ? 1 : 0);
            add_member_int(vsapi, out, "compiler_available", device.get_info<CL_DEVICE_COMPILER_AVAILABLE>() ? 1 : 0);
            add_member_int(vsapi, out, "linker_available", device.get_info<CL_DEVICE_LINKER_AVAILABLE>() ? 1 : 0);

            add_member_str(vsapi, out, "opencl_c_version", device.get_info<CL_DEVICE_OPENCL_C_VERSION>());

            add_member_int(
                vsapi, out, "image_max_buffer_size", device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)
            );
        },
        NULL, plugin
    );
}
