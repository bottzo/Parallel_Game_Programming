@echo off
pushd .
clang++ -std=c++14 "..\src\WindowsEngine.cpp"  "..\src\Engine.res" -DNDEBUG -DINITGUID -I"..\src\Dependencies\D3D12\include" -I"../src/Dependencies/pix/Include" -l"..\src\Dependencies\pix\bin\x64\WinPixEventRuntime.lib" -lkernel32.lib -luser32.lib -fno-cxx-exceptions -fno-exceptions -fno-stack-protector -mno-stack-arg-probe -mstack-probe-size=9999999 -fuse-ld=lld -g -Xlinker -nodefaultlib -Xlinker -subsystem:windows -Xlinker -stack:0x100000,0x100000
popd