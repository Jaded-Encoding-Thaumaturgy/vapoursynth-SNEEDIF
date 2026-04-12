# Setsugen No Ensemble of Edge Directed Interpolation Functions

Like [NNEDI3CL](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL) but made by Setsugennoao.
<img src="https://i.slow.pics/MnF0rcri.webp" height="25"/>

## NNEDI3

```python
sneedif.NNEDI3(clip, int field[, bint dh=False, bint dw=False, int[] planes=[0, 1, 2], int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, bint transpose_first=False, int device=-1])
```

- `transpose_first`: Transpose the clip before processing.

## Device Information

These functions return information about the available OpenCL environment and devices.

### ListDevices

```python
sneedif.ListDevices()
```

Returns a dictionary:

- `numDevices`: `int` — Total number of devices found.
- `deviceNames`: `str[]` — List of device names.
- `platformNames`: `str[]` — List of platform names.

### PlatformInfo

```python
sneedif.PlatformInfo([int device=-1])
```

Returns a dictionary for the specified `device`:

- `name`: `str`
- `vendor`: `str`
- `version`: `str`
- `profile`: `str`

### DeviceInfo

```python
sneedif.DeviceInfo([int device=-1])
```

Returns a dictionary containing detailed specifications for the specified `device`:

- `name`, `vendor`, `profile`, `version`, `opencl_c_version`: `str`
- `max_compute_units`, `max_work_group_size`, `image2D_max_width`, `image2D_max_height`: `int`
- `max_work_item_sizes`: `int[]`
- `image_support`, `available`, `compiler_available`, `linker_available`: `int` (boolean)
- `global_memory_cache_type`, `local_memory_type`: `str`
- `global_memory_cache`, `global_memory_size`, `max_constant_buffer_size`, `max_constant_arguments`, `local_memory_size`, `image_max_buffer_size`: `int`

## Installation

```bash
pip install vapoursynth-sneedif
```

_Note: Only wheels for Linux and Windows 64-bit are provided._

## Compilation

### Windows

Requirements:

- [MSYS2](https://www.msys2.org/)

1. Open **MSYS2 UCRT64** terminal.
2. Install dependencies:

   ```bash
   pacman -S mingw-w64-ucrt-x86_64-{cmake,meson,ninja,pkgconf,toolchain,boost,opencl-headers,opencl-icd,uv}
   ```

3. Build the wheel:

   ```bash
   uv build --wheel
   ```

### Linux

Requirements:

- [uv](https://github.com/astral-sh/uv).

1. Install dependencies:

   ```bash
   # Fedora / RHEL
   dnf install cmake gcc-c++ boost-devel opencl-headers ocl-icd-devel

   # Ubuntu / Debian
   apt install cmake g++ libboost-all-dev opencl-headers ocl-icd-opencl-dev
   ```

2. Build the wheel:

   ```bash
   uv build --wheel
   ```
