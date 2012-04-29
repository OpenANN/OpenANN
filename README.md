OpenANN
=======

An open source library for artificial neural networks.

License
-------

The license is LGPL 3. You can find the license text in the files `COPYING` and
`COPYING.LESSER`.

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