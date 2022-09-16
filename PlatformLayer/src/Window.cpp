RECT windowClientRect;

//included before renderer
struct D3D12_VIEWPORT;
struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12DescriptorHeap;
struct IDXGISwapChain4;
struct ID3D12CommandQueue;
struct ID3D12Fence;
struct DirectXRenderStruct;
void OnResize(int width, int height, bool contentLoaded, DirectXRenderStruct& renderer);

typedef struct WinProcInfo {
	bool* appRunning;
	bool* contentLoaded;
	DirectXRenderStruct* renderer;
}WinProcInfo;

inline WinProcInfo* GetAppState(HWND hwnd)
{
	return (WinProcInfo*)GetWindowLongPtr(hwnd, GWLP_USERDATA);;
}

LRESULT CALLBACK Win32WindowProc(
	HWND   hwnd,
	UINT   uMsg,
	WPARAM wParam,
	LPARAM lParam
)
{
	WinProcInfo* pState = GetAppState(hwnd);

	switch (uMsg)
	{
	case WM_CREATE: 
	{
		CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
		pState = reinterpret_cast<WinProcInfo*>(pCreate->lpCreateParams);
		SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)pState);
		return 0; 
	}
	case WM_SIZE:
	{
		////RESIZE THE BITMAP
		//RECT clientRect;
		GetClientRect(hwnd, &windowClientRect);	//x,y del rect son sempre 0
		int width = ((int)(short)LOWORD(lParam));
		int height = ((int)(short)HIWORD(lParam));
		OnResize(width, height, *pState->contentLoaded, *pState->renderer);
		//Win32ResizeDIBSection(&globalBackBuffer, clientRect.right, clientRect.bottom);
		return 0; //we return 0 when we have processed the message
	}
	case WM_DESTROY:
		//Todo: Handle this as an error - recreate window?
		*pState->appRunning = false;
		return 0;
	case WM_CLOSE:
		//Todo: Handle with message to user
		*pState->appRunning = false;
		return 0;
	case WM_ACTIVATEAPP:
		return 0;
	case WM_SYSKEYDOWN:
	case WM_SYSKEYUP:
	case WM_KEYDOWN:
	case WM_KEYUP:
	{
		bool wasDown = (lParam & ((1 << 30) != 0));
		bool isDown = (lParam & ((1 << 31) == 0));
		//if(wasDown != isDown) //no key repeat
		switch (wParam)
		{
		case 'W':
			break;
		case 'A':
			break;
		case 'S':
			break;
		case 'D':
			break;
		case VK_UP:
			break;
		case VK_LEFT:
			break;
		case VK_DOWN:
			break;
		case VK_RIGHT:
			break;
		case VK_ESCAPE:
			*pState->appRunning = false;
			break;
		case VK_F4:
			//TODO: maybe handle syskeys on the syskeys message with the DefWindowProc()
			if (lParam & ((1 << 29) != 0)) //if alt pressed
				*pState->appRunning = false;
		case VK_SPACE:
			break;
		default:
			break;
		}
		return 0;
	}
	default:
		return DefWindowProcA(hwnd, uMsg, wParam, lParam);
	}
}

HWND InitWindow(HINSTANCE hInstance, WinProcInfo* procInfo)
{
	WNDCLASSEXA windowClass = {};
	windowClass.cbSize = sizeof(WNDCLASSEXA);
	windowClass.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = Win32WindowProc;
	windowClass.hInstance = hInstance;
	HICON wndIcon = (HICON)LoadImageA(hInstance, MAKEINTRESOURCE(IDI_ICON1), IMAGE_ICON, 0, 0, LR_DEFAULTCOLOR);
	if (wndIcon)
		windowClass.hIcon = wndIcon;
	windowClass.lpszClassName = "Engine Window";
	if (!RegisterClassEx(&windowClass))
	{
		//TODO: error creating the window
		OutputDebugString("Error creating the window \nThe app will close");
		ExitProcess(-1);
	}
	HWND windowHandle = CreateWindowExA(
		0,
		windowClass.lpszClassName,
		"Engine",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		0,
		0,
		hInstance,
		procInfo);

	//TODO: cal setejar la clientRect aqui??
	GetClientRect(windowHandle, &windowClientRect);
	return windowHandle;
}

inline void ProcessWindowMessages(bool* appRunning)
{
	//process window messages
	MSG message;
	while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE))
	{
		if (message.message == WM_QUIT)
			*appRunning = false;

		TranslateMessage(&message);
		DispatchMessageA(&message);
	}
}