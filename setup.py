import os
import sys
import re
import warnings
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as orig_build_ext
from setuptools.command.build_clib import build_clib as orig_build_clib
from setuptools.command.install_lib import install_lib as orig_install_lib
from setuptools.command.develop import develop as orig_develop
from setuptools.command.bdist_egg import bdist_egg as orig_bdist_egg

from Cython.Build import cythonize


AD3_FLAGS_UNIX = [
    '-std=c++11',
    '-O3',
    '-Wall',
    '-Wno-sign-compare',
    '-Wno-overloaded-virtual',
    '-c',
    '-fmessage-length=0',
    '-fPIC',
    '-ffast-math',
    '-march=native'
]


AD3_FLAGS_MSVC = [
    '/O2',
    '/fp:fast',
    '/favor:INTEL64',
    '/wd4267'  # suppress sign-compare--like warning
]


AD3_CFLAGS =  {
    'cygwin' : AD3_FLAGS_UNIX,
    'mingw32' : AD3_FLAGS_UNIX,
    'unix' : AD3_FLAGS_UNIX,
    'msvc' : AD3_FLAGS_MSVC
}


# thanks to https://github.com/dfm/transit/blob/master/setup.py
def find_eigen():
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = []
    env_var = os.getenv("EIGEN_DIR")
    if env_var:
        search_dirs.append(env_var)
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    warnings.warn("Could not find eigen. Please set EIGEN_DIR.")
    return None


# support compiler-specific cflags in extensions and libs
class build_ext(orig_build_ext):
    def build_extensions(self):

        # bug in distutils: flag not valid for c++
        flag = '-Wstrict-prototypes'
        if (hasattr(self.compiler, 'compiler_so')
                and flag in self.compiler.compiler_so):
            self.compiler.compiler_so.remove(flag)

        compiler_type = self.compiler.compiler_type
        compile_args = AD3_CFLAGS.get(compiler_type, [])

        for e in self.extensions:
            e.extra_compile_args.extend(compile_args)

        # build_clib = self.get_finalized_command("build_clib")
        # print(build_clib.build_clib)

        super().build_extensions()


class build_clib(orig_build_clib):

    # output ad3 library to lpsmap/lib/libad3
    def initialize_options(self):
        super().initialize_options()
        build_py = self.get_finalized_command('build_py')
        output_dir = os.path.join(build_py.get_package_dir("lpsmap"),
                                  "core", "lib")

        self.build_clib = output_dir

    def build_libraries(self, libraries):
        # bug in distutils: flag not valid for c++
        flag = '-Wstrict-prototypes'
        if (hasattr(self.compiler, 'compiler_so')
                and flag in self.compiler.compiler_so):
            self.compiler.compiler_so.remove(flag)

        compiler_type = self.compiler.compiler_type
        compile_args = AD3_CFLAGS.get(compiler_type, [])

        for (lib_name, build_info) in libraries:
            build_info['cflags'] = compile_args

        super().build_libraries(libraries)


class install_lib(orig_install_lib):
    def install(self):
        # need to make sure that libad3 is in the build dir before installing
        # file is expected in

        build_clib = self.get_finalized_command('build_clib')

        # not trying to be generic, seek specifically "libad3*".
        libname = "libad3"
        # Code can be made generic using build_clib.libraries.

        lib_path = build_clib.build_clib
        tgt = os.path.join(self.build_dir, lib_path)
        self.mkpath(tgt)
        with os.scandir(lib_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.startswith(libname):
                    self.copy_file(entry.path, os.path.join(tgt, entry.name))

        super().install()


# this is a backport of a workaround for a problem in distutils.
# install_lib doesn't call build_clib
class bdist_egg(orig_bdist_egg):
    def run(self):
        # self.call_command('build_clib')
        bdist_egg.run(self)


class develop(orig_develop):
    def install_for_development(self):
        print("triggering clib")
        self.run_command('build_clib')
        super().install_for_development()


cmdclass = {
    'build_ext': build_ext,
    'build_clib': build_clib,
    'bdist_egg': bdist_egg,
    'develop': develop,
    'install_lib': install_lib,
}


libad3 = ('ad3', {
    'language': "c++",
    'sources': ['ad3qp/ad3/FactorGraph.cpp',
                'ad3qp/ad3/GenericFactor.cpp',
                'ad3qp/ad3/Factor.cpp',
                'ad3qp/ad3/Utils.cpp',
                ],
    'include_dirs': ['.',
                     './ad3qp',
                     find_eigen()
                     ],
})


extensions = [
    Extension('lpsmap.ad3qp.factor_graph', ["lpsmap/ad3qp/factor_graph.pyx"]),
    Extension('lpsmap.ad3qp.base', ["lpsmap/ad3qp/base.pyx"]),
    Extension('lpsmap.ad3ext.sequence',
              ["lpsmap/ad3ext/sequence.pyx"]),
    Extension('lpsmap.ad3ext.tree',
              ["lpsmap/ad3ext/tree.pyx",
               "lpsmap/ad3ext/DependencyDecoder.cpp"]),
]

setup(name='lp-sparsemap',
      version="0.9",
      libraries=[libad3],
      author="Vlad Niculae",
      packages=['lpsmap', 'lpsmap.api'],
      install_requires=["numpy>=1.14.6"],
      extras_require={'torch': 'torch>=1.8.1'},
      package_data={'lpsmap': ['ad3qp/*.pxd', 'core/lib/*', 'core/include/ad3/*']},
      cmdclass=cmdclass,
      include_package_data=True,
      ext_modules=cythonize(extensions)
)
