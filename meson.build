project(
  'SNEEDIF',
  'cpp',
  default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++20'],
  meson_version: '>=1.10.0',
  version: '4.3',
)

add_project_arguments(
  '-ffast-math',
  '-DNNEDI3_DATADIR="@0@"'.format(join_paths(get_option('prefix'), get_option('datadir'), meson.project_name())),
  language: 'cpp',
)
add_project_arguments(
  '-Wno-unused-variable',
  '-Wno-ignored-attributes',
  '-Wno-sign-compare',
  '-Wno-narrowing',
  '-Wno-int-in-bool-context',
  '-ffast-math',
  language: 'cpp',
)
add_project_arguments('-D_WIN32_WINNT=0x0A00', language: 'cpp')  # Target Win10+
add_project_arguments('-DCL_TARGET_OPENCL_VERSION=120', language: 'cpp')

sources = [
  'SNEEDIF/SNEEDIF.cpp',
  'SNEEDIF/NNEDI3.cpp',
]

py = import('python').find_installation(pure: false)

r = run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true)
inc_vs = include_directories(r.stdout().strip())

boost_dep = dependency('boost', modules: ['filesystem'], static: host_machine.system() != 'linux')

if host_machine.system() == 'darwin'
  opencl_dep = dependency('appleframeworks', modules: ['OpenCL'])
else
  opencl_dep = dependency('OpenCL', required: false)
  if not opencl_dep.found()
    opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
  endif
endif

deps = [boost_dep, opencl_dep]

if host_machine.system() == 'windows'
  deps += [meson.get_compiler('cpp').find_library('winpthread', static: true)]
endif

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language: 'cpp')
endif

shared_module(
  'sneedif',
  sources,
  dependencies: deps,
  include_directories: inc_vs,
  install: false,
  gnu_symbol_visibility: 'hidden',
)
