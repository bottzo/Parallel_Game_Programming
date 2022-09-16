cmake_minimum_required(VERSION "3.13")

project(Engine)

#NOTE: This could be set manually. If you change the path have in mind that the output binary should be on the same directory as the dlls and the D3D12 folder
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../build/ CACHE PATH "Where to output the final executable")
if(MSVC)
#don't add the release and debug folders after the directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set_directory_properties(PROPERTIES VS_STARTUP_PROJECT ${PROJECT_NAME})

set(SOURCE_FILES
src/Audio.cpp
src/COM.cpp
src/FileSystem.cpp
src/Input.cpp
src/Renderer.cpp
src/Textures.cpp 
src/Utils.cpp 
src/Window.cpp 
)
#no va pk els files nomes s'inclouen i no es posen com a source del projecte
source_group("Source Files" FILES ${SOURCE_FILES})
set(SHADER_FILES 
src/shaders/GenerateMips_CS.hlsl
src/shaders/PixelShader.hlsl
src/shaders/VertexShader.hlsl
)
source_group("Shader Files" FILES ${SHADER_FILES})
endif(MSVC)

add_executable(${PROJECT_NAME} WIN32 src/WindowsEngine.cpp) #src/Renderer.cpp)
#Correct way to add .rc files???
#https://discourse.cmake.org/t/how-to-use-resource-files-rc-in-cmake/2628
target_sources(${PROJECT_NAME} PRIVATE src/Engine.rc)
target_include_directories(${PROJECT_NAME} BEFORE PRIVATE src/Dependencies/D3D12/include src/Dependencies/pix/Include) #src/Dependencies/DirectXTex/include)
#if i don't define INITGUID all the GUID from DEFINE_GUID() macro used in wincoded.h evaluate to null and then don't work on release mode
add_compile_definitions(INITGUID)
target_link_directories(${PROJECT_NAME} PRIVATE src/Dependencies/pix/bin/x64/)#src/Dependencies/DirectXTex/lib/Release/)
#TODO: treure el pix profiler al fer les release builds
target_link_libraries(${PROJECT_NAME} PRIVATE kernel32.lib User32.lib windowscodecs.lib WinPixEventRuntime.lib) #DXGI.lib d3d12.lib DirectXTex.lib)  windowscodecs.lib 
#not sure if the fno-stack-protecktor is working (check)
#TODO: disable runtime error chacks from debug code (/RTC) $<$<CONFIG:DEBUG>:/RTC>
#TODO: /EHa en release pk si no error al linking de _std_terminate (s'ha de mirar una mica lo de les exceptions')
target_compile_options(${PROJECT_NAME} PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:DEBUG>>>:/EHa /GR-> $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:DEBUG>>>:-fno-cxx-exceptions -fno-exceptions --no-standard-libraries -fno-builtin -DNDEBUG>   $<$<CXX_COMPILER_ID:MSVC>: -GS- -Gs9999999> $<$<CXX_COMPILER_ID:Clang>:-fuse-ld=lld -fno-stack-protector -mstack-probe-size=9999999 -mno-stack-arg-probe>)
#error clang: cmake omits a repeated flag (=> it just puts the first -Xlinker and omits the rest causing it to crush the compilation)!!!!
target_link_options(${PROJECT_NAME} PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:DEBUG>>>:/NODEFAULTLIB> $<$<CXX_COMPILER_ID:MSVC>:-stack:0x100000,0x100000> $<$<CXX_COMPILER_ID:Clang>:--for-linker -stack:0x100000,0x100000> $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:DEBUG>>>:-Xlinker -nodefaultlib>)

#TODO
#C++ STD version
#Multiplatform (chack if windows or linux)