cmake_minimum_required(VERSION 3.7)
project(CarLicensePlateDetection)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp LPRegistrar.cpp LPRegistrar.h )

#include_directories(${OpenCV_LIBS})
find_package( OpenCV REQUIRED )
find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )


add_executable(CarLicensePlateDetection ${SOURCE_FILES})

target_link_libraries(CarLicensePlateDetection ${OpenCV_LIBS} curl json-c ${BOOST_LIBRARIES})
