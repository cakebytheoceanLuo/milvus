FetchContent_Declare(google_benchmark
    URL             https://github.com/google/benchmark/archive/v1.5.3.tar.gz
    URL_MD5         abb43ef7784eaf0f7a98aed560920f46
    DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH}
    SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/google_benchmark
    BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/google_benchmark
        )

FetchContent_GetProperties( google_benchmark )
if ( NOT google_benchmark_POPULATED )

    FetchContent_Populate( google_benchmark )

    # Adding the following targets:
    # gtest, gtest_main, gmock, gmock_main
    message("gb=${google_benchmark_SOURCE_DIR}")
    add_subdirectory( ${google_benchmark_SOURCE_DIR}
                      ${google_benchmark_BINARY_DIR}
                      EXCLUDE_FROM_ALL )
endif()
