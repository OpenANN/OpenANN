OpenANN
=======

An open source library for artificial neural networks.

License
-------

The license is LGPL 3. You can find the license text in the files `COPYING` and
`COPYING.LESSER`. Note that you can activate the CMake option
`USE_GPL_LIBRARIES` in order to use the library ALGLIB which is necessary to use
the optimization algorithms conjugate gradient and Levenberg-Marquardt. In this
case the compiled shared library libopenann is under GPL license.

Requirements
------------

* CMake 2.8 or higher
* C++ compiler, e. g. g++
* build management tool that is supported by CMake, e. g. make
* Eigen 3 library

Installation
------------

Linux

    cd path/to/OpenANN/dir
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE:String=Release ..
    sudo make install
    sudo ldconfig

Guidelines for Developers
-------------------------

* TODO link to todo list
* TODO link to coding style
* Use the namespace OpenANN.

Directory Structure
-------------------

* OpenANN - Header files that are exposed to the user
* benchmarks - Standard ANN benchmarks
* cmake - CMake macros
* doc - Additional doxygen documentation files
* examples - OpenANN showcase
* lib - External libraries
* ruby - Ruby language bindings
* src - Implementation files for libopenann
* test - Contains the OpenANN test suite
