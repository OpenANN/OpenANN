try:
    import build_info
except ImportError:
    import os
    import subprocess
    import warnings
    warnings.warn("No build information available. You must call CMake first.")
    cmake_dir = os.path.join("..", "build")
    if not os.path.exists(cmake_dir):
        os.makedirs(cmake_dir)
    subprocess.call("cd %s; cmake ..; make install" % cmake_dir)


if __name__ == "__main__":
    import os
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    import numpy

    extra_compile_args = build_info.compile_args
    extra_compile_args.append("-Wno-enum-compare")

    setup(
        name = 'OpenANN',
        version = build_info.version,
        description = "Python bindings for OpenANN.",
        author = "Alexander Fabisch",
        author_email = "afabisch@googlemail.com",
        url = "https://github.com/AlexanderFabisch/OpenANN",
        platforms = ["Linux", "MacOS"],
        license = "GPL",
        ext_modules=[
            Extension(
                "openann",
                sources=["cbindings.pxd", "openann.pyx", "evaluation.pxi",
                         "net.pxi", "optimization.pxi", "util.pxi",
                         "dataset.pxi", "layer.pxi", "preprocessing.pxi"],
                library_dirs=[
                    os.path.join(build_info.project_bin_dir, "lib"),
                    os.path.join(build_info.project_bin_dir, "src")],
                libraries=build_info.libraries,
                include_dirs=[
                    numpy.get_include(),
                    build_info.project_src_dir,
                    os.path.join(build_info.project_src_dir, "lib", "ALGLIB"),
                    os.path.join(build_info.project_src_dir, "test", "lib",
                                 "CPP-Test"),
                    build_info.eigen_inc],
                define_macros=[("NDEBUG",)],
                extra_compile_args=extra_compile_args,
                language="c++"),
          ],
        cmdclass = {'build_ext': build_ext},
    )
