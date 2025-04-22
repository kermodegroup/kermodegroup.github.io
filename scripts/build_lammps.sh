#!/bin/bash
cd mylammps
​
mkdir build
​
cd build


cmake ../cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
               -D PKG_PYTHON=yes -D PKG_ML-PACE=yes -D PKG_OPENMP=yes \
               -D BUILD_MPI=yes \

​
cmake --build . -j 40
​
cmake --install .
​
make install-python

cd ..
​
rm -r -d build
​
cd ..
​

echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib64:$LD_LIBRARY_PATH' >> $VIRTUAL_ENV/bin/activate
echo 'export LAMMPS_POTENTIALS=$VIRTUAL_ENV/share/lammps/potentials/' >> $VIRTUAL_ENV/bin/activate