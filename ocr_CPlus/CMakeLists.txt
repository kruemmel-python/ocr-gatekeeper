cmake_minimum_required(VERSION 3.16)
project(ocr_gatekeeper_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)

add_executable(ocr_gatekeeper_cpp src/main.cpp)

target_include_directories(ocr_gatekeeper_cpp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TESSERACT_INCLUDE_DIRS}
)

target_link_libraries(ocr_gatekeeper_cpp PRIVATE
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    Threads::Threads
)

target_compile_options(ocr_gatekeeper_cpp PRIVATE
    ${TESSERACT_CFLAGS_OTHER}
)
