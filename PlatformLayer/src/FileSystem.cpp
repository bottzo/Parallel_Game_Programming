typedef struct  {
	LONG distanceToMove;
	PLONG pDistanceToMoveHigh;
	DWORD startingPoint;
}FilePointer;

HRESULT ReadFileToBuffer(const char* filePath, void** buffer, DWORD* bufferSize, FilePointer fPtr = { 0, NULL, FILE_BEGIN })
{
	//TODO: manage better when error with reading file (NOT ALL PATHS WITH ERRORS CLOSE THE FILE!!!)
	HRESULT hr = S_OK;
	HANDLE hFile = CreateFile(filePath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, fPtr.distanceToMove, fPtr.pDistanceToMoveHigh, fPtr.startingPoint)) {
		CloseHandle(hFile);
		return HRESULT_FROM_WIN32(GetLastError());
	}
	LARGE_INTEGER fileSize;
	if (!GetFileSizeEx(hFile, &fileSize)) {
		CloseHandle(hFile);
		return HRESULT_FROM_WIN32(GetLastError());
	}
	//TODO: memory allocation
	*buffer = HeapAlloc(GetProcessHeap(), HEAP_NO_SERIALIZE, fileSize.QuadPart);
	if (0 == ReadFile(hFile, *buffer, fileSize.LowPart, bufferSize, NULL))
		hr = HRESULT_FROM_WIN32(GetLastError());
	CloseHandle(hFile);
	return hr;
}

HRESULT WriteBufferToFile(const char* filePath, void* buffer, DWORD bufferSize, FilePointer fPtr = { 0, NULL, FILE_BEGIN })
{
	//TODO: manage better when error with reading file (NOT ALL PATHS WITH ERRORS CLOSE THE FILE!!!)
	HRESULT hr = S_OK;
	HANDLE hFile = CreateFile(filePath, GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, 0, NULL);
	if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, fPtr.distanceToMove, fPtr.pDistanceToMoveHigh, fPtr.startingPoint)) {
		CloseHandle(hFile);
		return HRESULT_FROM_WIN32(GetLastError());
	}
	if (0 == WriteFile(hFile, buffer, bufferSize, NULL, NULL))
		hr = HRESULT_FROM_WIN32(GetLastError());
	CloseHandle(hFile);
	return hr;
}

bool FileExists(const char* path)
{
	DWORD dwAttrib = GetFileAttributes(path);
	return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool DirectoryExists(const char* path)
{
	DWORD dwAttrib = GetFileAttributes(path);
	return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

unsigned long long FileLastWriteTime(const char* path)
{
	WIN32_FILE_ATTRIBUTE_DATA fileInfo = {};
	DWORD dwAttrib = GetFileAttributesEx(path,_GET_FILEEX_INFO_LEVELS::GetFileExInfoStandard, &fileInfo);
	if (dwAttrib == INVALID_FILE_ATTRIBUTES)
		return 0;
	FILETIME fileTime = fileInfo.ftLastWriteTime;
	ULARGE_INTEGER ret = {};
	ret.LowPart = fileTime.dwLowDateTime;
	ret.HighPart = fileTime.dwHighDateTime;
	return ret.QuadPart;
}

//PATH UTILS --------------------------------------------------------------------------------

char* PointFilenameFromPath(char* path) {
	char* ret = path;
	while (*path != '\0')
	{
		if (*path == '/' || *path == '\\')
			ret = path + 1;
		++path;
	}
	return ret;
}

char* PointExtensionFromPath(char* path) {
	char* ret = path;
	while (*path != '\0')
	{
		if (*path == '.')
			ret = path + 1;
		++path;
	}
	return ret;
}