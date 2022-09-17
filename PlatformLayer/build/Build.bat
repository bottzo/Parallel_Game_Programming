#This batch file expects the clang compiler to be on the current directory or on the environment variables
@echo off
pushd .
clang++ -std=c++14 "..\src\WindowsEngine.cpp" "..\src\Engine.res" -DNDEBUG -DINITGUID -I"../src/Dependencies/pix/Include" -I"../src/Dependencies/D3D12/include" -l"..\src\Dependencies\pix\bin\x64\WinPixEventRuntime.lib" -lkernel32.lib -luser32.lib --no-standard-libraries -fno-builtin -fno-cxx-exceptions -fno-exceptions -fno-stack-protector -mno-stack-arg-probe -mstack-probe-size=9999999 -fuse-ld=lld -O3 -Xlinker -nodefaultlib -Xlinker -subsystem:windows -Xlinker -stack:0x100000,0x100000
popd