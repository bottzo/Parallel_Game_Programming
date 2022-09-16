#include <windows.h>

#include "resource1.h"

//#define PROFILE
#include "pix3.h"

#include "Utils.cpp"
#include "Window.cpp"
#include "Input.cpp"
#include "COM.cpp"
#include "Audio.cpp"
#include "FileSystem.cpp"
#include "Textures.cpp"
#include "Renderer.cpp"

#ifndef _DEBUG
//no CRT  problems
//The compiler will generate the code expecting the CRT to be there so it expects the following func and variables to exist
extern "C"
{
	//TODO: disable runtime checks  to compile!!!!
	//TODO: problemes amb les exceptions !!!!!
	//TODO: if i don't define __CxxFrameHandler4 i have to disable exceptions (C/C++->Code Generation->enable cpp exceptions)
	int __CxxFrameHandler4;
	int _fltused;
//#ifdef _DEBUG
//pragma function because the compiler has reserved the memset from the CRT to be an intrinsic so we have to tell it to take this one instead of the CRT  one that is not there
#pragma function(memset)
	void* memset(void* dest, int c, size_t count)
	{
		char* bytes = (char*)dest;
		while (count--)
		{
			*bytes++ = (char)c;
		}
		return dest;
	}

#pragma function(memcpy)
	void* memcpy(void* dest, const void* src, size_t count)
	{
		char* dest8 = (char*)dest;
		const char* src8 = (const char*)src;
		while (count--)
		{
			*dest8++ = *src8++;
		}
		return dest;
	}
#pragma function(memcmp)
	int memcmp(const void* ptr1, const void* ptr2, size_t num)
	{
		const unsigned char* one = (const unsigned char*)ptr1;
		const unsigned char* two = (const unsigned char*)ptr2;
		unsigned int i = 0;
		while (num--)
		{
			if (one[i] != two[i])
			{
				if (one[i] < two[i])
					return 1;
				else
					return -1;
			}
			++i;
		}
		return 0;
	}
//#endif
}
#endif

//WinMain to pass ansi cmd and wWinMain to pass Unicode cmd
//int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow)
{	
	//GetCommandLine();
	//SetThreadPriority(hInstance, THREAD_PRIORITY_HIGHEST);
	bool appRunning = true;
	bool contentLoaded = false;

	bool ComInitialized = Win32InitializeCOM();
	InitInput();
	AudioStruct audio = {};
	InitAudio(audio,ComInitialized);
	IWICImagingFactory* iFactory = nullptr;
	InitTextures(reinterpret_cast<LPVOID*>(&iFactory), ComInitialized);
	//Init renderer---------------------------------------------------------------------------------------------
	DirectXRenderStruct renderer = {};
	InitRenderer(renderer);
	LoadShaderCompiler();

	WinProcInfo winProcInfo = {&appRunning, &contentLoaded, &renderer};
	HWND windowHandle = InitWindow(hInstance, &winProcInfo);


	renderer.sChain = CreateSwapChain(windowHandle, renderer.cQueue.iCQueue, 0, 0, 3);
	ID3D12Resource* backBuffers[3] = {};
	renderer.backBuffers = backBuffers;
	renderer.backBufferCount = _countof(backBuffers);
	renderer.rtvDescHeap = CreateRTVDescriptorHeap(renderer.device, renderer.backBufferCount, renderer.sChain, renderer.backBuffers, renderer.currentBackBufferIndex);
	renderer.tearingSupport = CheckTearingSupport();
	
	RenderLoadedAssets loadedAssets = LoadAssets(renderer.device, renderer.cQueue.iCQueue, iFactory, &contentLoaded);

	ResizeDepthBuffer(renderer.device, &renderer.depthBuffer, renderer.dsvDescHeap, windowClientRect.right, windowClientRect.bottom, contentLoaded, renderer.cQueue);

	//NOTE: can't use class destructors on a global variable because they need the crt call to atexit function !!!
	//Containers::MyQueue<ID3D12GraphicsCommandList2*> cListQueue;
	//Containers::MyQueue<syncAllocator> cAllocatorQueue;
	//---------------------------------------------------------------------------------------------------------

	long long performanceFrequency;
	LARGE_INTEGER perfFreq;
	QueryPerformanceFrequency(&perfFreq);
	performanceFrequency = perfFreq.QuadPart;
	LARGE_INTEGER lastCounter;
	QueryPerformanceCounter(&lastCounter);
	unsigned long long lastCycleCount = __rdtsc();

	while (appRunning)
	{
		PIXScopedEvent(PIX_COLOR_INDEX(2), "Frame");
		ProcessWindowMessages(&appRunning);

		//poll controller states
		for (DWORD controllerIndex = 0; controllerIndex < XUSER_MAX_COUNT; ++controllerIndex)
		{
			XINPUT_STATE controllerState;
			if (XInputGetState(controllerIndex, &controllerState) == ERROR_SUCCESS /*ERROR_DEVICE_NOT_CONNECTED*/)
			{
				//This controller is plugged in
				//TODO: some inputs missing
				XINPUT_GAMEPAD* pad = &controllerState.Gamepad;

				bool up = (pad->wButtons & XINPUT_GAMEPAD_DPAD_UP);
				bool down = (pad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
				bool left = (pad->wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
				bool right = (pad->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
				bool start = (pad->wButtons & XINPUT_GAMEPAD_START);
				bool back = (pad->wButtons & XINPUT_GAMEPAD_BACK);
				bool leftShoulder = (pad->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
				bool rightShoulder = (pad->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
				bool aButton = (pad->wButtons & XINPUT_GAMEPAD_A);
				bool bButton = (pad->wButtons & XINPUT_GAMEPAD_B);
				bool xButton = (pad->wButtons & XINPUT_GAMEPAD_X);
				bool yButton = (pad->wButtons & XINPUT_GAMEPAD_Y);

				short stickX = pad->sThumbLX;
				short stickY = pad->sThumbLY;

				//TODO: handle the deadzone apropietly
				//#define XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  7849
				//#define XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE 8689
				//xOffset += stickX / 4096; //4096 = 2^12
				//yOffset += stickY / 4096;
			}
			//else
			//{
			//	//controller not aviable
			//}
		}

		XINPUT_VIBRATION vibration;
		vibration.wLeftMotorSpeed = 6000;
		vibration.wRightMotorSpeed = 6000;
		XInputSetState(0, &vibration);

		//DEL RENDERER
		OnUpdate();
		OnRender(loadedAssets, renderer);

		unsigned long long endCycleCount = __rdtsc();
		LARGE_INTEGER endCounter;
		QueryPerformanceCounter(&endCounter);
		long long counterElapsed = endCounter.QuadPart - lastCounter.QuadPart;
		unsigned long long cyclesElapsed = endCycleCount - lastCycleCount;
		float msPerFrame = 1000 * (float)counterElapsed / (float)performanceFrequency;
		float fps = (float)performanceFrequency / (float)counterElapsed;
		float mCyclesPerFrame = ((float)cyclesElapsed / (1000 * 1000));
#ifdef _DEBUG
		char buffer[128];
		sprintf(buffer, "Miliseconds/Frame: %f FPS: %f megacycles/frame: %f\n", msPerFrame, fps, mCyclesPerFrame);
		OutputDebugStringA(buffer);
#else
		//char* charMsPerFrame = ConvertFloatToChar(msPerFrame);
		//OutputDebugStringA(charMsPerFrame);
		//HeapFree(GetProcessHeap(), 0, charMsPerFrame);
		//OutputDebugStringA("ms \n");
#endif // _DEBUG

		lastCounter = endCounter;
		lastCycleCount = endCycleCount;
	}
	//TODO: Fer CleanUps ??? (no els necesitem si el modul esta viu tot el programa)
	//delete buffer del audio ???
	//we are leaking memory just releasing the interfaces!!!! (renderer clean up)
	//adapter->Release(); device->Release(); cQueue->Release(); sChain->Release(); dHeap->Release(); cAllocator->Release(); ...
	if (audio.pDataBuffer)
		VirtualFree(audio.pDataBuffer, audio.dwChunkSize, MEM_RELEASE);
		//HeapFree(GetProcessHeap(), 0, pDataBuffer);
}

//WinMainCRTStartup to pass ansi cmd and wWinMainCRTStartup to pass Unicode cmd calling wWinMain or WinMain
#ifndef _DEBUG
void __stdcall WinMainCRTStartup()
{
	int Result = WinMain(GetModuleHandle(0), 0, 0, 0);
	ExitProcess(Result);
}
#endif