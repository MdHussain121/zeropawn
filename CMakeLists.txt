cmake_minimum_required(VERSION 3.18)
project(zeropawn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CTest integration
include(CTest)

# Static-link the C++ runtime on MinGW so .pyd / .exe don't need external DLLs
if(MINGW)
    set(_STATIC_FLAGS "-static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_STATIC_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_STATIC_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${_STATIC_FLAGS}")
endif()

# ── Options ──────────────────────────────────────────────────────────
option(ZEROPAWN_BUILD_TESTS   "Build C++ tests"          ON)
option(ZEROPAWN_BUILD_PYTHON  "Build pybind11 module"    ON)

# Backward-compatibility for legacy option names
if(DEFINED CHESSRL_BUILD_TESTS AND NOT DEFINED ZEROPAWN_BUILD_TESTS)
    set(ZEROPAWN_BUILD_TESTS ${CHESSRL_BUILD_TESTS})
endif()
if(DEFINED CHESSRL_BUILD_PYTHON AND NOT DEFINED ZEROPAWN_BUILD_PYTHON)
    set(ZEROPAWN_BUILD_PYTHON ${CHESSRL_BUILD_PYTHON})
endif()

# ── Fetch pybind11 ───────────────────────────────────────────────────
if(ZEROPAWN_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Subdirectories ───────────────────────────────────────────────────
add_subdirectory(cpp)
