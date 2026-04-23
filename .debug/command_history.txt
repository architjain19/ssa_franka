=================================== FRANKA ROS SETUP ===================================

----------------------------------- VERSION / BRANCH ---------------------------------
# gcc version = 13.3.0
# python version = 3.9.18
# FCI version = 4.0.4
# libfranka version = 0.8.0
# franka_ros branch = 0.8.0
# libfranka branch = 0.8.0
# serl_franka_controllers branch = main
--------------------------------------------------------------------------------------

conda activate  # will activate (base)
conda config --show channels
conda config --env --remove channels https://repo.anaconda.com/pkgs/main
conda config --env --remove channels https://repo.anaconda.com/pkgs/r
conda config --env --remove channels defaults
conda create -n env_franka python=3.9 ros-noetic-desktop -c robostack -c conda-forge
conda activate env_franka
conda config --env --add channels robostack-noetic
conda deactivate
conda activate env_franka
conda install -c conda-forge ros-dev-tools
roscore     # runs rosmaster
gcc --version   # was greater than '13.3.0'
conda install -c conda-forge gcc=13.3.0
gcc --version   # should be '13.3.0'
which python    # should be from 'miniconda' dir
which gcc   # should be from 'miniconda' dir
python --version    # should be '3.9.x' mostly '3.9.18'
mkdir -p ~/archit/ssa_ws/src && cd ~/archit/ssa_ws/src
catkin_init_workspace src
git clone --recursive https://github.com/frankarobotics/franka_ros src/franka_ros
conda deactivate
conda activate env_franka
roscore     # check if rosmaster is running to make sure ros is installed properly
rosdep install --from-paths src --ignore-src --rosdistro noetic -y --skip-keys "libfranka franka_gazebo"

========================================= NOTES ==========================================
- DO NOT install libfranka using conda or conda-forge/robostack channels.
- Neither using ros-noetic-libfranka (ex: conda install ros-noetic-libfranka -c robostack -c conda-forge)
==========================================================================================

conda install -c conda-forge eigen poco
cd ~/archit/ssa_ws/
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.8.0

cd ~/archit/ssa_ws/src/franka_ros
git checkout 0.8.0

cd ~/archit/ssa_ws/libfranka
mkdir build && cd build

------------------------------------ CODE CHANGES ------------------------------------
sed -i '5a #include <stdexcept>' ~/archit/ssa_ws/libfranka/src/control_types.cpp
sed -i '6a #include <string>' ~/archit/ssa_ws/libfranka/include/franka/control_tools.h
sed -i '12a #include <cstdint>' ~/archit/ssa_ws/src/franka_ros/franka_hw/include/franka_hw/resource_helpers.h
--------------------------------------------------------------------------------------

cd ~/archit/ssa_ws/libfranka/build
cmake --build . -j$(nproc)
cmake --install .

cd ~/archit/ssa_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
========================================================================================



============================= SERL FRANKA CONTROLLER SETUP =============================
cd ~/archit/ssa_ws/src
git clone --recursive https://github.com/rail-berkeley/serl_franka_controllers.git
cd ..
source ~/archit/ssa_ws/devel/setup.zsh
export PATH="/home/daphne/miniconda3/envs/env_franka/bin:$PATH"\n
rospack find franka_control


# Check what's currently in the CMakeLists\n
cat >> ~/archit/ssa_ws/src/serl_franka_controllers/CMakeLists.txt << 'EOF'

------------------ CODE CHANGES ------------------
# Force C++14 to fix std::allocator::rebind compatibility with ROS Noetic
set_target_properties(serl_franka_controllers PROPERTIES CXX_STANDARD 14)
target_compile_options(serl_franka_controllers PRIVATE -std=c++14)
EOF
--------------------------------------------------

cd ~/archit/ssa_ws\ncatkin_make --pkg serl_franka_controllers \\n  -DCMAKE_BUILD_TYPE=Release \\n  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \\n  -DCMAKE_CXX_FLAGS="-include cstdint -include stdexcept -include string" \\n  -Dfranka_DIR=$CONDA_PREFIX/lib/cmake/Franka
cd ~/archit/ssa_ws\ncatkin_make -DCMAKE_BUILD_TYPE=Release \\n  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \\n  -DCMAKE_CXX_FLAGS="-include cstdint -include stdexcept -include string" \\n  -Dfranka_DIR=$CONDA_PREFIX/lib/cmake/Franka
source devel/setup.zsh
export PATH="/home/daphne/miniconda3/envs/env_franka/bin:$PATH"
========================================================================================

# TO ACTIVATE AND SOURCE THE ENVIRONMENT
conda activate env_franka && source ~/archit/ssa_ws/devel/setup.zsh && export PATH="/home/daphne/miniconda3/envs/env_franka/bin:$PATH"