#include <xaudio2.h>

typedef HRESULT CreateXAudio2(IXAudio2** ppXAudio2, UINT32 Flags, XAUDIO2_PROCESSOR XAudio2Processor);
CreateXAudio2* Win32LoadXAudio2()
{
	HMODULE hXaudio2Library = LoadLibraryA("XAUDIO2_9.DLL");
	if (!hXaudio2Library)
		hXaudio2Library = LoadLibraryA("XAUDIO2_8.DLL");
	if (hXaudio2Library)
		return (CreateXAudio2*)GetProcAddress(hXaudio2Library, "XAudio2Create");
	return nullptr;
}

#ifdef _XBOX //Big-Endian
#define fourccRIFF 'RIFF'
#define fourccDATA 'data'
#define fourccFMT 'fmt '
#define fourccWAVE 'WAVE'
#define fourccXWMA 'XWMA'
#define fourccDPDS 'dpds'
#endif

#ifndef _XBOX //Little-Endian
#define fourccRIFF 'FFIR'
#define fourccDATA 'atad'
#define fourccFMT ' tmf'
#define fourccWAVE 'EVAW'
#define fourccXWMA 'AMWX'
#define fourccDPDS 'sdpd'
#endif
HRESULT FindChunk(HANDLE hFile, DWORD fourcc, DWORD& dwChunkSize, DWORD& dwChunkDataPosition)
{
	HRESULT hr = S_OK;
	if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, 0, NULL, FILE_BEGIN))
		return HRESULT_FROM_WIN32(GetLastError());

	DWORD dwChunkType;
	DWORD dwChunkDataSize;
	DWORD dwRIFFDataSize = 0;
	DWORD dwFileType;
	DWORD bytesRead = 0;
	DWORD dwOffset = 0;

	while (hr == S_OK)
	{
		DWORD dwRead;
		if (0 == ReadFile(hFile, &dwChunkType, sizeof(DWORD), &dwRead, NULL))
			hr = HRESULT_FROM_WIN32(GetLastError());

		if (0 == ReadFile(hFile, &dwChunkDataSize, sizeof(DWORD), &dwRead, NULL))
			hr = HRESULT_FROM_WIN32(GetLastError());

		switch (dwChunkType)
		{
		case fourccRIFF:
			dwRIFFDataSize = dwChunkDataSize;
			dwChunkDataSize = 4;
			if (0 == ReadFile(hFile, &dwFileType, sizeof(DWORD), &dwRead, NULL))
				hr = HRESULT_FROM_WIN32(GetLastError());
			break;

		default:
			if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, dwChunkDataSize, NULL, FILE_CURRENT))
				return HRESULT_FROM_WIN32(GetLastError());
		}

		dwOffset += sizeof(DWORD) * 2;

		if (dwChunkType == fourcc)
		{
			dwChunkSize = dwChunkDataSize;
			dwChunkDataPosition = dwOffset;
			return S_OK;
		}

		dwOffset += dwChunkDataSize;

		if (bytesRead >= dwRIFFDataSize)
			return S_FALSE;

	}

	return S_OK;

}

HRESULT ReadChunkData(HANDLE hFile, void* buffer, DWORD buffersize, DWORD bufferoffset)
{
	HRESULT hr = S_OK;
	if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, bufferoffset, NULL, FILE_BEGIN))
		return HRESULT_FROM_WIN32(GetLastError());
	DWORD dwRead;
	if (0 == ReadFile(hFile, buffer, buffersize, &dwRead, NULL))
		hr = HRESULT_FROM_WIN32(GetLastError());
	return hr;
}

typedef struct AudioStruct {
	IXAudio2* pXAudio2;
	IXAudio2MasteringVoice* pMasterVoice;
	BYTE* pDataBuffer;
	DWORD dwChunkSize;
	XAUDIO2_BUFFER buffer;
	IXAudio2SourceVoice* pSourceVoice;
	WAVEFORMATEXTENSIBLE wfx;
}AudioStruct;

inline void InitAudio(AudioStruct& audio, bool COMInitialized)
{
	//audio struct:
	//IXAudio2* pXAudio2 = nullptr;
	//IXAudio2MasteringVoice* pMasterVoice = nullptr;
	//BYTE* pDataBuffer = nullptr;
	//DWORD dwChunkSize;
	//XAUDIO2_BUFFER buffer;
	//IXAudio2SourceVoice* pSourceVoice;
	//WAVEFORMATEXTENSIBLE wfx;

	//Falta el CoUninitialize() !!!!
	//TODO: com manejo la initialitzacio de COM
	if (COMInitialized)//SUCCEEDED(CoInitializeExPtr(nullptr, COINIT_MULTITHREADED)))
	{
		CreateXAudio2* createAudio = Win32LoadXAudio2();
		if (createAudio)
		{
			//IXAudio2* pXAudio2 = nullptr;
			if (SUCCEEDED(createAudio(&audio.pXAudio2, 0, XAUDIO2_DEFAULT_PROCESSOR)))
				//IXAudio2MasteringVoice* pMasterVoice = nullptr;
				if (FAILED(audio.pXAudio2->CreateMasteringVoice(&audio.pMasterVoice)))
					OutputDebugStringA("Failed creating mastering voice");
		}
	}
	if (audio.pXAudio2 && audio.pMasterVoice)
	{
		//NOMES LLEGIM WAV => fer servir audacity per transformar els fitxers a .wav
		HANDLE hFile = CreateFile("../assets/mino1.wav", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
		if (INVALID_HANDLE_VALUE != hFile)
		{
			if (SetFilePointer(hFile, 0, NULL, FILE_BEGIN) != INVALID_SET_FILE_POINTER)
			{
				//DWORD dwChunkSize;
				DWORD dwChunkPosition;
				//check the file type, should be fourccWAVE or 'XWMA'
				FindChunk(hFile, fourccRIFF, audio.dwChunkSize, dwChunkPosition);
				DWORD filetype;
				ReadChunkData(hFile, &filetype, sizeof(DWORD), dwChunkPosition);
				if (filetype == fourccWAVE)
				{
					//WAVEFORMATEXTENSIBLE wfx;
					FindChunk(hFile, fourccFMT, audio.dwChunkSize, dwChunkPosition);
					ReadChunkData(hFile, &audio.wfx, audio.dwChunkSize, dwChunkPosition);

					//fill out the audio data buffer with the contents of the fourccDATA chunk
					FindChunk(hFile, fourccDATA, audio.dwChunkSize, dwChunkPosition);
					//BYTE* pDataBuffer = new BYTE[dwChunkSize];
					audio.pDataBuffer = (BYTE*)VirtualAlloc(NULL, audio.dwChunkSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
					//pDataBuffer = (BYTE*)HeapAlloc(GetProcessHeap(), 0, dwChunkSize);
					ReadChunkData(hFile, audio.pDataBuffer, audio.dwChunkSize, dwChunkPosition);

					//XAUDIO2_BUFFER buffer;
					audio.buffer.AudioBytes = audio.dwChunkSize;  //size of the audio buffer in bytes
					audio.buffer.pAudioData = audio.pDataBuffer;  //buffer containing audio data
					audio.buffer.Flags = XAUDIO2_END_OF_STREAM; // tell the source voice not to expect any data after this buffer

					//IXAudio2SourceVoice* pSourceVoice;
					if (SUCCEEDED(audio.pXAudio2->CreateSourceVoice(&audio.pSourceVoice, (WAVEFORMATEX*)&audio.wfx)) && SUCCEEDED(audio.pSourceVoice->SubmitSourceBuffer(&audio.buffer)))
					{
						if (FAILED(audio.pSourceVoice->Start(0)))
							OutputDebugString("too bad");
					}
				}
			}
			CloseHandle(hFile);
		}
	}
}