OpenANN
=======

An open source library for artificial neural networks.

License
-------

The license is GPL 3. You can find the license text in the files `COPYING`.

Minimum Requirements
--------------------

* CMake 2.8 or higher
* C++ compiler, e. g. g++
* build management tool that is supported by CMake, e. g. make
* Eigen 3 library
* shell, wget, unzip

Installation
------------

Linux

    cd path/to/OpenANN/dir
    mkdir build
    cd build
    # Available CMAKE_BUILD_TYPEs are Debug and Release.
    cmake -D CMAKE_BUILD_TYPE:String=Release ..
    sudo make install
    sudo ldconfig

Directory Structure
-------------------

* OpenANN - Header files that are exposed to the user
* benchmarks - Standard ANN benchmarks
* cmake - CMake macros
* doc - Additional doxygen documentation files
* examples - OpenANN showcase
* lib - External libraries
* python - Python language bindings
* src - Implementation files for libopenann
* test - Contains the OpenANN test suite
