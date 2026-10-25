# ── Sources ──────────────────────────────────────────────────────────
set(ZEROPAWN_SOURCES
    src/stub.cpp
    src/chess_game.cpp
    src/encoder.cpp
    src/mcts.cpp
    src/connect4.cpp
)

# ── Static library ───────────────────────────────────────────────────
add_library(zeropawn_core STATIC ${ZEROPAWN_SOURCES})
target_include_directories(zeropawn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/chess-library
)
target_compile_features(zeropawn_core PUBLIC cxx_std_17)

# ── pybind11 module ──────────────────────────────────────────────────
if(ZEROPAWN_BUILD_PYTHON)
    pybind11_add_module(zeropawn_cpp src/bindings.cpp)
    target_link_libraries(zeropawn_cpp PRIVATE zeropawn_core)
endif()

# ── C++ tests ────────────────────────────────────────────────────────
if(ZEROPAWN_BUILD_TESTS)
    add_executable(zeropawn_tests tests/test_main.cpp)
    target_link_libraries(zeropawn_tests PRIVATE zeropawn_core)
    add_test(NAME zeropawn_unit_tests COMMAND zeropawn_tests)
endif()
