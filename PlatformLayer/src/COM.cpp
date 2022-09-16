typedef HRESULT MyCoInitializeEx(LPVOID pvReserved, DWORD dwCoInit);
typedef HRESULT MyCoCreateInstance(REFCLSID  rclsid,LPUNKNOWN pUnkOuter,DWORD dwClsContext,REFIID riid,LPVOID* ppv);
typedef void MyPropVariantClear(PROPVARIANT* pvar);

static MyCoInitializeEx* CoInitializeExPtr = nullptr;
static MyCoCreateInstance* CoCreateInstancePtr = nullptr;
static MyPropVariantClear* PropVariantClearPtr = nullptr;
static IID MY_GUID_NULL;

bool Win32InitializeCOM()
{
	HMODULE hOle32Library = LoadLibraryA("Ole32.dll");
	if (hOle32Library)
	{
		CoInitializeExPtr = (MyCoInitializeEx*)GetProcAddress(hOle32Library, "CoInitializeEx");
		CoCreateInstancePtr = (MyCoCreateInstance*)GetProcAddress(hOle32Library, "CoCreateInstance");
		PropVariantClearPtr = (MyPropVariantClear*)GetProcAddress(hOle32Library, "PropVariantClear");
		memset(&MY_GUID_NULL, 0, sizeof(IID));
		return SUCCEEDED(CoInitializeExPtr(nullptr, COINIT_MULTITHREADED));
	}
	return false;
}

//SUCCEEDED(initCom(nullptr, COINIT_MULTITHREADED));