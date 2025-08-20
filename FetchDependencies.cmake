include(FetchContent)

# GTest - Makes me feel good to include it
if (NOT TARGET googletest)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()

# Event management. Useful for state update notifications
if (NOT TARGET cpp-events)
    FetchContent_Declare(
            cpp-event
            GIT_REPOSITORY https://github.com/kayoscode/cpp-event.git
    )

    FetchContent_MakeAvailable(cpp-event)
endif ()