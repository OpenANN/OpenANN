if __name__ == "__main__":
  from distutils.core import setup
  from distutils.extension import Extension
  from Cython.Distutils import build_ext

  setup(
    name = 'libopenann',
    ext_modules=[
      Extension("openann",
                sources=["openann.pyx"],
                library_dirs=["../build/src",
                              "../build/lib"],
                libraries=["openann",
                           "openann_cuda",
                           "alglib"],
                include_dirs=[".",
                              "..",
                              "../OpenANN",
                              "/usr/include/eigen3",
                              "../test/lib/CPP-Test"], # TODO configure
                define_macros=[("fpt", "float"), ("Vt", "Eigen::VectorXf"), ("Mt", "Eigen::MatrixXf"), ("PARALLEL_CORES", "4"), ("CUDA_AVAILABLE",), ("USE_GPL_LICENSE",), ("NDEBUG",)],
                extra_compile_args=["-g0", "-s", "-O3", "-msse", "-msse2", "-msse3", "-mssse3", "-msse4.1", "-msse4.2", "-fopenmp"], # TODO configure
                language="c++"),
      ],
    cmdclass = {'build_ext': build_ext},
  )
