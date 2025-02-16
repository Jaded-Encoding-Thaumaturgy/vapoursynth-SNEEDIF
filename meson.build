project('SNEEDIF', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++20'],
  meson_version : '>=0.51.0',
  version : '8'
)

add_project_arguments('-ffast-math', '-DNNEDI3_DATADIR="@0@"'.format(join_paths(get_option('prefix'), get_option('datadir'), meson.project_name())), language : 'cpp')
add_project_arguments('-Wno-unused-variable', '-Wno-ignored-attributes', '-Wno-sign-compare', '-Wno-narrowing', '-Wno-int-in-bool-context', '-ffast-math', language : 'cpp')


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

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

boost_dep = dependency('boost', modules : ['filesystem'], static : true)

opencl_dep = dependency('OpenCL', required : false)
if not opencl_dep.found()
  opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
endif

deps = [vapoursynth_dep, boost_dep, opencl_dep]

if host_machine.system() == 'windows'
  deps += [meson.get_compiler('cpp').find_library('winpthread', static: true)]
endif

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
endif

shared_module('sneedif', sources,
  dependencies : deps,
  gnu_symbol_visibility : 'hidden'
)
