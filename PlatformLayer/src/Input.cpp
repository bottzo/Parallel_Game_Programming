#include <Xinput.h>

//xInput
//needed functions from xinput (XInputGetState // XInputSetState)
//linked at runtime to avoid problems with windows versions that don't support the xinput library
#define X_INPUT_GET_STATE(name) DWORD WINAPI name(DWORD dwUserIndex, XINPUT_STATE* pState)
typedef X_INPUT_GET_STATE(ControllerGetState);
X_INPUT_GET_STATE(UnlinkedXInputGetState)
{
	return ERROR_DEVICE_NOT_CONNECTED;
}
static ControllerGetState* DyXInputGetState = UnlinkedXInputGetState;

#define X_INPUT_SET_STATE(name) DWORD WINAPI name(DWORD dwUserIndex,XINPUT_VIBRATION* pVibration)
typedef X_INPUT_SET_STATE(ControllerSetState);
X_INPUT_SET_STATE(UnlinkedXInputSetState)
{
	return ERROR_DEVICE_NOT_CONNECTED;
}
static ControllerSetState* DyXInputSetState = UnlinkedXInputSetState;

#define XInputGetState DyXInputGetState
#define XInputSetState DyXInputSetState

inline void InitInput()
{
	HMODULE hXInputLibrary = LoadLibraryA("XInput1_4.dll");
	if (!hXInputLibrary)
		hXInputLibrary = LoadLibraryA("XInput9_1_0.dll");
	else if (!hXInputLibrary)
		hXInputLibrary = LoadLibraryA("xinput1_3.dll");

	if (hXInputLibrary)
	{
		DyXInputGetState = (ControllerGetState*)GetProcAddress(hXInputLibrary, "XInputGetState");
		DyXInputSetState = (ControllerSetState*)GetProcAddress(hXInputLibrary, "XInputSetState");
	}
}