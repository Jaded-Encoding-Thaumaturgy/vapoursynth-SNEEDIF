project('SNEEDIF', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++20'],
  meson_version : '>=1.10.0',
  version : '4.2'
)

add_project_arguments('-ffast-math', '-DNNEDI3_DATADIR="@0@"'.format(join_paths(get_option('prefix'), get_option('datadir'), meson.project_name())), language : 'cpp')
add_project_arguments('-Wno-unused-variable', '-Wno-ignored-attributes', '-Wno-sign-compare', '-Wno-narrowing', '-Wno-int-in-bool-context', '-ffast-math', language : 'cpp')
add_project_arguments('-D_WIN32_WINNT=0x0A00', language: 'cpp') # Target Win10+
add_project_arguments('-DCL_TARGET_OPENCL_VERSION=200', language: 'cpp') # Explicitly target OpenCL 2.0

sources = [
  'SNEEDIF/SNEEDIF.cpp',
  'SNEEDIF/NNEDI3.cpp',
  'SNEEDIF/half_float/umHalf.h',
  'SNEEDIF/vectorclass/instrset_detect.cpp',
  'SNEEDIF/vectorclass/instrset.h',
  'SNEEDIF/vectorclass/vectorclass.h',
  'SNEEDIF/vectorclass/vectorf128.h',
  'SNEEDIF/vectorclass/vectorf256.h',
  'SNEEDIF/vectorclass/vectorf256e.h',
  'SNEEDIF/vectorclass/vectorf512.h',
  'SNEEDIF/vectorclass/vectorf512e.h',
  'SNEEDIF/vectorclass/vectori128.h',
  'SNEEDIF/vectorclass/vectori256.h',
  'SNEEDIF/vectorclass/vectori256e.h',
  'SNEEDIF/vectorclass/vectori512.h',
  'SNEEDIF/vectorclass/vectori512e.h'
]

py = import('python').find_installation(pure: false)

r = run_command(
    py,
    '-c',
    'import vapoursynth as vs; print(vs.get_include())',
    check: true,
)
inc_vs = include_directories(r.stdout().strip())

boost_dep = dependency('boost', modules : ['filesystem'], static : host_machine.system() == 'windows')

opencl_dep = dependency('OpenCL', required : false)
if not opencl_dep.found()
  opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
endif

deps = [boost_dep, opencl_dep]

if host_machine.system() == 'windows'
  deps += [meson.get_compiler('cpp').find_library('winpthread', static: true)]
endif

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
endif

shared_module('sneedif', sources,
  dependencies : deps,
  include_directories : inc_vs,
  install: true,
  install_dir: py.get_install_dir() / 'vapoursynth/plugins',
  gnu_symbol_visibility : 'hidden'
)
