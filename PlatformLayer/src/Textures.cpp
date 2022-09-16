#include <assert.h>
#include <wincodec.h>

//IWICImagingFactory* g_Factory = nullptr;
//bool gWIC2 = false; we are presuming that we have wic 2 // WIC2 is available on Windows 10, Windows 8.x, and Windows 7 SP1 with KB 2670838 installed
inline bool InitTextures(LPVOID* ifactory, bool COMInitialized)
{
	if (COMInitialized)
	{
        //HMODULE hwindowscodecs = LoadLibraryA("windowscodecs.dll");
        //if (!hwindowscodecs)
        //    return false;

        HRESULT hr = CoCreateInstancePtr(
            CLSID_WICImagingFactory2,
            nullptr,
            CLSCTX_INPROC_SERVER,
            __uuidof(IWICImagingFactory2),
            ifactory
        );
        if (SUCCEEDED(hr))
            return TRUE;
	}
    return false;
}

//DIRECXTexture (with some modifications)-------------------------------------------------------------------------------------------------------------------
enum TEX_DIMENSION
    // Subset here matches D3D10_RESOURCE_DIMENSION and D3D11_RESOURCE_DIMENSION
{
    TEX_DIMENSION_TEXTURE1D = 2,
    TEX_DIMENSION_TEXTURE2D = 3,
    TEX_DIMENSION_TEXTURE3D = 4,
};

enum WIC_FLAGS
{
    WIC_FLAGS_NONE = 0x0,

    WIC_FLAGS_FORCE_RGB = 0x1,
    // Loads DXGI 1.1 BGR formats as DXGI_FORMAT_R8G8B8A8_UNORM to avoid use of optional WDDM 1.1 formats

    WIC_FLAGS_NO_X2_BIAS = 0x2,
    // Loads DXGI 1.1 X2 10:10:10:2 format as DXGI_FORMAT_R10G10B10A2_UNORM

    WIC_FLAGS_NO_16BPP = 0x4,
    // Loads 565, 5551, and 4444 formats as 8888 to avoid use of optional WDDM 1.2 formats

    WIC_FLAGS_ALLOW_MONO = 0x8,
    // Loads 1-bit monochrome (black & white) as R1_UNORM rather than 8-bit grayscale

    WIC_FLAGS_ALL_FRAMES = 0x10,
    // Loads all images in a multi-frame file, converting/resizing to match the first frame as needed, defaults to 0th frame otherwise

    WIC_FLAGS_IGNORE_SRGB = 0x20,
    // Ignores sRGB metadata if present in the file

    WIC_FLAGS_DITHER = 0x10000,
    // Use ordered 4x4 dithering for any required conversions

    WIC_FLAGS_DITHER_DIFFUSION = 0x20000,
    // Use error-diffusion dithering for any required conversions

    WIC_FLAGS_FILTER_POINT = 0x100000,
    WIC_FLAGS_FILTER_LINEAR = 0x200000,
    WIC_FLAGS_FILTER_CUBIC = 0x300000,
    WIC_FLAGS_FILTER_FANT = 0x400000, // Combination of Linear and Box filter
        // Filtering mode to use for any required image resizing (only needed when loading arrays of differently sized images; defaults to Fant)
};

struct TexMetadata
{
    unsigned long long          width;
    unsigned long long          height;     // Should be 1 for 1D textures
    unsigned long long          depth;      // Should be 1 for 1D or 2D textures
    unsigned long long          arraySize;  // For cubemap, this is a multiple of 6
    unsigned long long          mipLevels;
    unsigned int        miscFlags;
    unsigned int        miscFlags2;
    DXGI_FORMAT     format;
    TEX_DIMENSION   dimension;
};

// Bitmap image container
struct Image
{
    size_t      width;
    size_t      height;
    DXGI_FORMAT format;
    size_t      rowPitch;
    size_t      slicePitch;
    unsigned char* pixels;
};

//-------------------------------------------------------------------------------------
    // WIC Pixel Format Translation Data
//-------------------------------------------------------------------------------------
struct WICTranslate
{
    GUID        wic;
    DXGI_FORMAT format;
    bool        srgb;
};

const WICTranslate g_WICFormats[] =
{
    { GUID_WICPixelFormat128bppRGBAFloat,       DXGI_FORMAT_R32G32B32A32_FLOAT,         false },

    { GUID_WICPixelFormat64bppRGBAHalf,         DXGI_FORMAT_R16G16B16A16_FLOAT,         false },
    { GUID_WICPixelFormat64bppRGBA,             DXGI_FORMAT_R16G16B16A16_UNORM,         true },

    { GUID_WICPixelFormat32bppRGBA,             DXGI_FORMAT_R8G8B8A8_UNORM,             true },
    { GUID_WICPixelFormat32bppBGRA,             DXGI_FORMAT_B8G8R8A8_UNORM,             true }, // DXGI 1.1
    { GUID_WICPixelFormat32bppBGR,              DXGI_FORMAT_B8G8R8X8_UNORM,             true }, // DXGI 1.1

    { GUID_WICPixelFormat32bppRGBA1010102XR,    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, true }, // DXGI 1.1
    { GUID_WICPixelFormat32bppRGBA1010102,      DXGI_FORMAT_R10G10B10A2_UNORM,          true },

    { GUID_WICPixelFormat16bppBGRA5551,         DXGI_FORMAT_B5G5R5A1_UNORM,             true },
    { GUID_WICPixelFormat16bppBGR565,           DXGI_FORMAT_B5G6R5_UNORM,               true },

    { GUID_WICPixelFormat32bppGrayFloat,        DXGI_FORMAT_R32_FLOAT,                  false },
    { GUID_WICPixelFormat16bppGrayHalf,         DXGI_FORMAT_R16_FLOAT,                  false },
    { GUID_WICPixelFormat16bppGray,             DXGI_FORMAT_R16_UNORM,                  true },
    { GUID_WICPixelFormat8bppGray,              DXGI_FORMAT_R8_UNORM,                   true },

    { GUID_WICPixelFormat8bppAlpha,             DXGI_FORMAT_A8_UNORM,                   false },

    { GUID_WICPixelFormatBlackWhite,            DXGI_FORMAT_R1_UNORM,                   false },
};
//-------------------------------------------------------------------------------------
    // WIC Pixel Format nearest conversion table
    //-------------------------------------------------------------------------------------

struct WICConvert
{
    GUID        source;
    GUID        target;
};

const WICConvert g_WICConvert[] =
{
    // Directly support the formats listed in XnaTexUtil::g_WICFormats, so no conversion required
    // Note target GUID in this conversion table must be one of those directly supported formats.

    { GUID_WICPixelFormat1bppIndexed,           GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat2bppIndexed,           GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat4bppIndexed,           GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat8bppIndexed,           GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 

    { GUID_WICPixelFormat2bppGray,              GUID_WICPixelFormat8bppGray }, // DXGI_FORMAT_R8_UNORM 
    { GUID_WICPixelFormat4bppGray,              GUID_WICPixelFormat8bppGray }, // DXGI_FORMAT_R8_UNORM 

    { GUID_WICPixelFormat16bppGrayFixedPoint,   GUID_WICPixelFormat16bppGrayHalf }, // DXGI_FORMAT_R16_FLOAT 
    { GUID_WICPixelFormat32bppGrayFixedPoint,   GUID_WICPixelFormat32bppGrayFloat }, // DXGI_FORMAT_R32_FLOAT 

    { GUID_WICPixelFormat16bppBGR555,           GUID_WICPixelFormat16bppBGRA5551 }, // DXGI_FORMAT_B5G5R5A1_UNORM 
    { GUID_WICPixelFormat32bppBGR101010,        GUID_WICPixelFormat32bppRGBA1010102 }, // DXGI_FORMAT_R10G10B10A2_UNORM

    { GUID_WICPixelFormat24bppBGR,              GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat24bppRGB,              GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat32bppPBGRA,            GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 
    { GUID_WICPixelFormat32bppPRGBA,            GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM 

    { GUID_WICPixelFormat48bppRGB,              GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat48bppBGR,              GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat64bppBGRA,             GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat64bppPRGBA,            GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat64bppPBGRA,            GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM

    { GUID_WICPixelFormat48bppRGBFixedPoint,    GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat48bppBGRFixedPoint,    GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat64bppRGBAFixedPoint,   GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat64bppBGRAFixedPoint,   GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat64bppRGBFixedPoint,    GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat64bppRGBHalf,          GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
    { GUID_WICPixelFormat48bppRGBHalf,          GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 

    { GUID_WICPixelFormat128bppPRGBAFloat,      GUID_WICPixelFormat128bppRGBAFloat }, // DXGI_FORMAT_R32G32B32A32_FLOAT 
    { GUID_WICPixelFormat128bppRGBFloat,        GUID_WICPixelFormat128bppRGBAFloat }, // DXGI_FORMAT_R32G32B32A32_FLOAT 
    { GUID_WICPixelFormat128bppRGBAFixedPoint,  GUID_WICPixelFormat128bppRGBAFloat }, // DXGI_FORMAT_R32G32B32A32_FLOAT 
    { GUID_WICPixelFormat128bppRGBFixedPoint,   GUID_WICPixelFormat128bppRGBAFloat }, // DXGI_FORMAT_R32G32B32A32_FLOAT 
    { GUID_WICPixelFormat32bppRGBE,             GUID_WICPixelFormat128bppRGBAFloat }, // DXGI_FORMAT_R32G32B32A32_FLOAT 

    { GUID_WICPixelFormat32bppCMYK,             GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM
    { GUID_WICPixelFormat64bppCMYK,             GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat40bppCMYKAlpha,        GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM
    { GUID_WICPixelFormat80bppCMYKAlpha,        GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM

#if (_WIN32_WINNT >= _WIN32_WINNT_WIN8) || defined(_WIN7_PLATFORM_UPDATE)
    { GUID_WICPixelFormat32bppRGB,              GUID_WICPixelFormat32bppRGBA }, // DXGI_FORMAT_R8G8B8A8_UNORM
    { GUID_WICPixelFormat64bppRGB,              GUID_WICPixelFormat64bppRGBA }, // DXGI_FORMAT_R16G16B16A16_UNORM
    { GUID_WICPixelFormat64bppPRGBAHalf,        GUID_WICPixelFormat64bppRGBAHalf }, // DXGI_FORMAT_R16G16B16A16_FLOAT 
#endif

    // We don't support n-channel formats
};

DXGI_FORMAT WICToDXGI(const GUID& guid)
{
    for (size_t i = 0; i < _countof(g_WICFormats); ++i)
    {
        if (memcmp(&g_WICFormats[i].wic, &guid, sizeof(GUID)) == 0)
            return g_WICFormats[i].format;
    }

#if (_WIN32_WINNT >= _WIN32_WINNT_WIN8) || defined(_WIN7_PLATFORM_UPDATE)
    //if (g_WIC2) //we are assuming we have wic2
    //{
    if (memcmp(&GUID_WICPixelFormat96bppRGBFloat, &guid, sizeof(GUID)) == 0)
        return DXGI_FORMAT_R32G32B32_FLOAT;
    //}
#endif

    return DXGI_FORMAT_UNKNOWN;
}

bool DXGIToWIC(DXGI_FORMAT format, GUID& guid, bool ignoreRGBvsBGR = false)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        if (ignoreRGBvsBGR)
        {
            // If we are not doing conversion so don't really care about BGR vs RGB color-order,
            // we can use the canonical WIC 32bppBGRA format which avoids an extra format conversion when using the WIC scaler
            memcpy(&guid, &GUID_WICPixelFormat32bppBGRA, sizeof(GUID));
        }
        else
        {
            memcpy(&guid, &GUID_WICPixelFormat32bppRGBA, sizeof(GUID));
        }
        return true;

    case DXGI_FORMAT_D32_FLOAT:
        memcpy(&guid, &GUID_WICPixelFormat32bppGrayFloat, sizeof(GUID));
        return true;

    case DXGI_FORMAT_D16_UNORM:
        memcpy(&guid, &GUID_WICPixelFormat16bppGray, sizeof(GUID));
        return true;

    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        memcpy(&guid, &GUID_WICPixelFormat32bppBGRA, sizeof(GUID));
        return true;

    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        memcpy(&guid, &GUID_WICPixelFormat32bppBGR, sizeof(GUID));
        return true;

#if (_WIN32_WINNT >= _WIN32_WINNT_WIN8) || defined(_WIN7_PLATFORM_UPDATE)
    case DXGI_FORMAT_R32G32B32_FLOAT:
        //if (g_WIC2)
        //{
            memcpy(&guid, &GUID_WICPixelFormat96bppRGBFloat, sizeof(GUID));
            return true;
        //}
        break;
#endif

    default:
        for (size_t i = 0; i < _countof(g_WICFormats); ++i)
        {
            if (g_WICFormats[i].format == format)
            {
                memcpy(&guid, &g_WICFormats[i].wic, sizeof(GUID));
                return true;
            }
        }
        break;
    }

    memcpy(&guid, &MY_GUID_NULL, sizeof(GUID));
    return false;
}

//-------------------------------------------------------------------------------------
// Converts to an SRGB equivalent type if available
//-------------------------------------------------------------------------------------
bool IsSRGBFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}

DXGI_FORMAT GetSRGBFormat(DXGI_FORMAT format)
{
    DXGI_FORMAT srgbFormat = format;
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        srgbFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        break;
    case DXGI_FORMAT_BC1_UNORM:
        srgbFormat = DXGI_FORMAT_BC1_UNORM_SRGB;
        break;
    case DXGI_FORMAT_BC2_UNORM:
        srgbFormat = DXGI_FORMAT_BC2_UNORM_SRGB;
        break;
    case DXGI_FORMAT_BC3_UNORM:
        srgbFormat = DXGI_FORMAT_BC3_UNORM_SRGB;
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        srgbFormat = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        break;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        srgbFormat = DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        break;
    case DXGI_FORMAT_BC7_UNORM:
        srgbFormat = DXGI_FORMAT_BC7_UNORM_SRGB;
        break;
    }

    return srgbFormat;
}

DXGI_FORMAT MakeSRGB(DXGI_FORMAT fmt)
{
    switch (fmt)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

    case DXGI_FORMAT_BC1_UNORM:
        return DXGI_FORMAT_BC1_UNORM_SRGB;

    case DXGI_FORMAT_BC2_UNORM:
        return DXGI_FORMAT_BC2_UNORM_SRGB;

    case DXGI_FORMAT_BC3_UNORM:
        return DXGI_FORMAT_BC3_UNORM_SRGB;

    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;

    case DXGI_FORMAT_B8G8R8X8_UNORM:
        return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;

    case DXGI_FORMAT_BC7_UNORM:
        return DXGI_FORMAT_BC7_UNORM_SRGB;

    default:
        return fmt;
    }
}

#define XBOX_DXGI_FORMAT_R10G10B10_7E3_A2_FLOAT DXGI_FORMAT(116)
#define XBOX_DXGI_FORMAT_R10G10B10_6E4_A2_FLOAT DXGI_FORMAT(117)
#define XBOX_DXGI_FORMAT_D16_UNORM_S8_UINT DXGI_FORMAT(118)
#define XBOX_DXGI_FORMAT_R16_UNORM_X8_TYPELESS DXGI_FORMAT(119)
#define XBOX_DXGI_FORMAT_X16_TYPELESS_G8_UINT DXGI_FORMAT(120)

#define WIN10_DXGI_FORMAT_P208 DXGI_FORMAT(130)
#define WIN10_DXGI_FORMAT_V208 DXGI_FORMAT(131)
#define WIN10_DXGI_FORMAT_V408 DXGI_FORMAT(132)

#ifndef XBOX_DXGI_FORMAT_R10G10B10_SNORM_A2_UNORM
#define XBOX_DXGI_FORMAT_R10G10B10_SNORM_A2_UNORM DXGI_FORMAT(189)
#endif

#define XBOX_DXGI_FORMAT_R4G4_UNORM DXGI_FORMAT(190)

//-------------------------------------------------------------------------------------
// Returns bits-per-pixel for a given DXGI format, or 0 on failure
//-------------------------------------------------------------------------------------
unsigned long long BitsPerPixel(DXGI_FORMAT fmt)
{
    switch (static_cast<int>(fmt))
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 128;

    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return 96;

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
    case DXGI_FORMAT_Y416:
    case DXGI_FORMAT_Y210:
    case DXGI_FORMAT_Y216:
        return 64;

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_R8G8_B8G8_UNORM:
    case DXGI_FORMAT_G8R8_G8B8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_AYUV:
    case DXGI_FORMAT_Y410:
    case DXGI_FORMAT_YUY2:
    case XBOX_DXGI_FORMAT_R10G10B10_7E3_A2_FLOAT:
    case XBOX_DXGI_FORMAT_R10G10B10_6E4_A2_FLOAT:
    case XBOX_DXGI_FORMAT_R10G10B10_SNORM_A2_UNORM:
        return 32;

    case DXGI_FORMAT_P010:
    case DXGI_FORMAT_P016:
    case XBOX_DXGI_FORMAT_D16_UNORM_S8_UINT:
    case XBOX_DXGI_FORMAT_R16_UNORM_X8_TYPELESS:
    case XBOX_DXGI_FORMAT_X16_TYPELESS_G8_UINT:
    case WIN10_DXGI_FORMAT_V408:
        return 24;

    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_A8P8:
    case DXGI_FORMAT_B4G4R4A4_UNORM:
    case WIN10_DXGI_FORMAT_P208:
    case WIN10_DXGI_FORMAT_V208:
        return 16;

    case DXGI_FORMAT_NV12:
    case DXGI_FORMAT_420_OPAQUE:
    case DXGI_FORMAT_NV11:
        return 12;

    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
    case DXGI_FORMAT_AI44:
    case DXGI_FORMAT_IA44:
    case DXGI_FORMAT_P8:
    case XBOX_DXGI_FORMAT_R4G4_UNORM:
        return 8;

    case DXGI_FORMAT_R1_UNORM:
        return 1;

    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        return 4;

    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return 8;

    default:
        return 0;
    }
}

enum TEX_FILTER_FLAGS
{
    TEX_FILTER_DEFAULT = 0,

    TEX_FILTER_WRAP_U = 0x1,
    TEX_FILTER_WRAP_V = 0x2,
    TEX_FILTER_WRAP_W = 0x4,
    TEX_FILTER_WRAP = (TEX_FILTER_WRAP_U | TEX_FILTER_WRAP_V | TEX_FILTER_WRAP_W),
    TEX_FILTER_MIRROR_U = 0x10,
    TEX_FILTER_MIRROR_V = 0x20,
    TEX_FILTER_MIRROR_W = 0x40,
    TEX_FILTER_MIRROR = (TEX_FILTER_MIRROR_U | TEX_FILTER_MIRROR_V | TEX_FILTER_MIRROR_W),
    // Wrap vs. Mirror vs. Clamp filtering options

    TEX_FILTER_SEPARATE_ALPHA = 0x100,
    // Resize color and alpha channel independently

    TEX_FILTER_FLOAT_X2BIAS = 0x200,
    // Enable *2 - 1 conversion cases for unorm<->float and positive-only float formats

    TEX_FILTER_RGB_COPY_RED = 0x1000,
    TEX_FILTER_RGB_COPY_GREEN = 0x2000,
    TEX_FILTER_RGB_COPY_BLUE = 0x4000,
    // When converting RGB to R, defaults to using grayscale. These flags indicate copying a specific channel instead
    // When converting RGB to RG, defaults to copying RED | GREEN. These flags control which channels are selected instead.

    TEX_FILTER_DITHER = 0x10000,
    // Use ordered 4x4 dithering for any required conversions
    TEX_FILTER_DITHER_DIFFUSION = 0x20000,
    // Use error-diffusion dithering for any required conversions

    TEX_FILTER_POINT = 0x100000,
    TEX_FILTER_LINEAR = 0x200000,
    TEX_FILTER_CUBIC = 0x300000,
    TEX_FILTER_BOX = 0x400000,
    TEX_FILTER_FANT = 0x400000, // Equiv to Box filtering for mipmap generation
    TEX_FILTER_TRIANGLE = 0x500000,
    // Filtering mode to use for any required image resizing

    TEX_FILTER_SRGB_IN = 0x1000000,
    TEX_FILTER_SRGB_OUT = 0x2000000,
    TEX_FILTER_SRGB = (TEX_FILTER_SRGB_IN | TEX_FILTER_SRGB_OUT),
    // sRGB <-> RGB for use in conversion operations
    // if the input format type is IsSRGB(), then SRGB_IN is on by default
    // if the output format type is IsSRGB(), then SRGB_OUT is on by default

    TEX_FILTER_FORCE_NON_WIC = 0x10000000,
    // Forces use of the non-WIC path when both are an option

    TEX_FILTER_FORCE_WIC = 0x20000000,
    // Forces use of the WIC path even when logic would have picked a non-WIC path when both are an option
};

#define TEX_FILTER_MASK 0xF00000
inline WICBitmapInterpolationMode __cdecl GetWICInterp(DWORD flags)
{
    static_assert(TEX_FILTER_POINT == 0x100000, "TEX_FILTER_ flag values don't match TEX_FILTER_MASK");

    static_assert(static_cast<int>(TEX_FILTER_POINT) == static_cast<int>(WIC_FLAGS_FILTER_POINT), "TEX_FILTER_* flags should match WIC_FLAGS_FILTER_*");
    static_assert(static_cast<int>(TEX_FILTER_LINEAR) == static_cast<int>(WIC_FLAGS_FILTER_LINEAR), "TEX_FILTER_* flags should match WIC_FLAGS_FILTER_*");
    static_assert(static_cast<int>(TEX_FILTER_CUBIC) == static_cast<int>(WIC_FLAGS_FILTER_CUBIC), "TEX_FILTER_* flags should match WIC_FLAGS_FILTER_*");
    static_assert(static_cast<int>(TEX_FILTER_FANT) == static_cast<int>(WIC_FLAGS_FILTER_FANT), "TEX_FILTER_* flags should match WIC_FLAGS_FILTER_*");

    switch (flags & TEX_FILTER_MASK)
    {
    case TEX_FILTER_POINT:
        return WICBitmapInterpolationModeNearestNeighbor;

    case TEX_FILTER_LINEAR:
        return WICBitmapInterpolationModeLinear;

    case TEX_FILTER_CUBIC:
        return WICBitmapInterpolationModeCubic;

    case TEX_FILTER_FANT:
    default:
        return WICBitmapInterpolationModeFant;
    }
}


unsigned long long CountMips(unsigned long long width, _In_ unsigned long long height)
{
    unsigned long long mipLevels = 1;

    while (height > 1 || width > 1)
    {
        if (height > 1)
            height >>= 1;

        if (width > 1)
            width >>= 1;

        ++mipLevels;
    }

    return mipLevels;
}

bool CalculateMipLevels(unsigned long long width, unsigned long long height, unsigned long long& mipLevels)
{
    if (mipLevels > 1)
    {
        unsigned long long maxMips = CountMips(width, height);
        if (mipLevels > maxMips)
            return false;
    }
    else if (mipLevels == 0)
    {
        mipLevels = CountMips(width, height);
    }
    else
    {
        mipLevels = 1;
    }
    return true;
}


//this flags are passed to the  flags argument on the ComputePitch()
enum CP_FLAGS
{
    CP_FLAGS_NONE = 0x0,      // Normal operation
    CP_FLAGS_LEGACY_DWORD = 0x1,      // Assume pitch is DWORD aligned instead of BYTE aligned
    CP_FLAGS_PARAGRAPH = 0x2,      // Assume pitch is 16-byte aligned instead of BYTE aligned
    CP_FLAGS_YMM = 0x4,      // Assume pitch is 32-byte aligned instead of BYTE aligned
    CP_FLAGS_ZMM = 0x8,      // Assume pitch is 64-byte aligned instead of BYTE aligned
    CP_FLAGS_PAGE4K = 0x200,    // Assume pitch is 4096-byte aligned instead of BYTE aligned
    CP_FLAGS_BAD_DXTN_TAILS = 0x1000,   // BC formats with malformed mipchain blocks smaller than 4x4
    CP_FLAGS_24BPP = 0x10000,  // Override with a legacy 24 bits-per-pixel format size
    CP_FLAGS_16BPP = 0x20000,  // Override with a legacy 16 bits-per-pixel format size
    CP_FLAGS_8BPP = 0x40000,  // Override with a legacy 8 bits-per-pixel format size
};

HRESULT ComputePitch(DXGI_FORMAT fmt, unsigned long long width, unsigned long long height,
    unsigned long long& rowPitch, unsigned long long& slicePitch, DWORD flags)
{
    unsigned long long pitch = 0;
    unsigned long long slice = 0;

    switch (static_cast<int>(fmt))
    {
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        //assert(IsCompressed(fmt));
        {
            if (flags & CP_FLAGS_BAD_DXTN_TAILS)
            {
                unsigned long long nbw = width >> 2;
                unsigned long long nbh = height >> 2;
                pitch = Max<unsigned long long>(1u, (unsigned long long)nbw * 8u);
                slice = Max<unsigned long long>(1u, pitch * (unsigned long long)nbh);
            }
            else
            {
                unsigned long long nbw = Max<unsigned long long>(1u, ((unsigned long long)width + 3u) / 4u);
                unsigned long long nbh = Max<unsigned long long>(1u, ((unsigned long long)height + 3u) / 4u);
                pitch = nbw * 8u;
                slice = pitch * nbh;
            }
        }
        break;

    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        //assert(IsCompressed(fmt));
        {
            if (flags & CP_FLAGS_BAD_DXTN_TAILS)
            {
                unsigned long long nbw = width >> 2;
                unsigned long long nbh = height >> 2;
                pitch = Max<unsigned long long>(1u, (unsigned long long)nbw * 16u);
                slice = Max<unsigned long long>(1u, pitch * (unsigned long long)nbh);
            }
            else
            {
                unsigned long long nbw = Max<unsigned long long>(1u, ((unsigned long long)width + 3u) / 4u);
                unsigned long long nbh = Max<unsigned long long>(1u, ((unsigned long long)height + 3u) / 4u);
                pitch = nbw * 16u;
                slice = pitch * nbh;
            }
        }
        break;

    case DXGI_FORMAT_R8G8_B8G8_UNORM:
    case DXGI_FORMAT_G8R8_G8B8_UNORM:
    case DXGI_FORMAT_YUY2:
        //assert(IsPacked(fmt));
        pitch = (((unsigned long long)width + 1u) >> 1) * 4u;
        slice = pitch * (unsigned long long)height;
        break;

    case DXGI_FORMAT_Y210:
    case DXGI_FORMAT_Y216:
        //assert(IsPacked(fmt));
        pitch = (((unsigned long long)width + 1u) >> 1) * 8u;
        slice = pitch * (unsigned long long)height;
        break;

    case DXGI_FORMAT_NV12:
    case DXGI_FORMAT_420_OPAQUE:
        //assert(IsPlanar(fmt));
        pitch = (((unsigned long long)width + 1u) >> 1) * 2u;
        slice = pitch * ((unsigned long long)height + (((unsigned long long)height + 1u) >> 1));
        break;

    case DXGI_FORMAT_P010:
    case DXGI_FORMAT_P016:
    case XBOX_DXGI_FORMAT_D16_UNORM_S8_UINT:
    case XBOX_DXGI_FORMAT_R16_UNORM_X8_TYPELESS:
    case XBOX_DXGI_FORMAT_X16_TYPELESS_G8_UINT:
        //assert(IsPlanar(fmt));
        pitch = (((unsigned long long)width + 1u) >> 1) * 4u;
        slice = pitch * ((unsigned long long)height + (((unsigned long long)height + 1u) >> 1));
        break;

    case DXGI_FORMAT_NV11:
        //assert(IsPlanar(fmt));
        pitch = (((unsigned long long)width + 3u) >> 2) * 4u;
        slice = pitch * (unsigned long long)height * 2u;
        break;

    case WIN10_DXGI_FORMAT_P208:
        //assert(IsPlanar(fmt));
        pitch = (((unsigned long long)width + 1u) >> 1) * 2u;
        slice = pitch * (unsigned long long)height * 2u;
        break;

    case WIN10_DXGI_FORMAT_V208:
        //assert(IsPlanar(fmt));
        pitch = (unsigned long long)width;
        slice = pitch * ((unsigned long long)height + ((((unsigned long long)height + 1u) >> 1) * 2u));
        break;

    case WIN10_DXGI_FORMAT_V408:
        //assert(IsPlanar(fmt));
        pitch = (unsigned long long)width;
        slice = pitch * ((unsigned long long)height + ((unsigned long long)(height >> 1) * 4u));
        break;

    default:
        //assert(!IsCompressed(fmt) && !IsPacked(fmt) && !IsPlanar(fmt));
        {
            unsigned long long bpp;

            if (flags & CP_FLAGS_24BPP)
                bpp = 24;
            else if (flags & CP_FLAGS_16BPP)
                bpp = 16;
            else if (flags & CP_FLAGS_8BPP)
                bpp = 8;
            else
                bpp = BitsPerPixel(fmt);

            if (!bpp)
                return E_INVALIDARG;

            if (flags & (CP_FLAGS_LEGACY_DWORD | CP_FLAGS_PARAGRAPH | CP_FLAGS_YMM | CP_FLAGS_ZMM | CP_FLAGS_PAGE4K))
            {
                if (flags & CP_FLAGS_PAGE4K)
                {
                    pitch = (((unsigned long long)width * bpp + 32767u) / 32768u) * 4096u;
                    slice = pitch * (unsigned long long)height;
                }
                else if (flags & CP_FLAGS_ZMM)
                {
                    pitch = (((unsigned long long)width * bpp + 511u) / 512u) * 64u;
                    slice = pitch * (unsigned long long)height;
                }
                else if (flags & CP_FLAGS_YMM)
                {
                    pitch = (((unsigned long long)width * bpp + 255u) / 256u) * 32u;
                    slice = pitch * (unsigned long long)height;
                }
                else if (flags & CP_FLAGS_PARAGRAPH)
                {
                    pitch = (((unsigned long long)width * bpp + 127u) / 128u) * 16u;
                    slice = pitch * (unsigned long long)height;
                }
                else // DWORD alignment
                {
                    // Special computation for some incorrectly created DDS files based on
                    // legacy DirectDraw assumptions about pitch alignment
                    pitch = (((unsigned long long)width * bpp + 31u) / 32u) * sizeof(unsigned int);
                    slice = pitch * (unsigned long long)height;
                }
            }
            else
            {
                // Default byte alignment
                pitch = ((unsigned long long)width * bpp + 7u) / 8u;
                slice = pitch * (unsigned long long)height;
            }
        }
        break;
    }

#if defined(_M_IX86) || defined(_M_ARM) || defined(_M_HYBRID_X86_ARM64)
    static_assert(sizeof(unsigned long long) == 4, "Not a 32-bit platform!");
    if (pitch > UINT32_MAX || slice > UINT32_MAX)
    {
        rowPitch = slicePitch = 0;
        return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
    }
#else
    static_assert(sizeof(unsigned long long) == 8, "Not a 64-bit platform!");
#endif

    rowPitch = static_cast<unsigned long long>(pitch);
    slicePitch = static_cast<unsigned long long>(slice);

    return S_OK;
}

//-------------------------------------------------------------------------------------
// Determines number of image array entries and pixel size
//-------------------------------------------------------------------------------------
bool DetermineImageArray(
    const TexMetadata& metadata,
    DWORD cpFlags,
    unsigned long long& nImages,
    unsigned long long& pixelSize)
{
    assert(metadata.width > 0 && metadata.height > 0 && metadata.depth > 0);
    assert(metadata.arraySize > 0);
    assert(metadata.mipLevels > 0);
    unsigned long long totalPixelSize = 0;
    unsigned long long nimages = 0;
    for (unsigned long long item = 0; item < metadata.arraySize; ++item)
    {
        unsigned long long w = metadata.width;
        unsigned long long h = metadata.height;

        for (unsigned long long level = 0; level < metadata.mipLevels; ++level)
        {
            unsigned long long rowPitch, slicePitch;
            if (FAILED(ComputePitch(metadata.format, w, h, rowPitch, slicePitch, cpFlags)))
            {
                nimages = totalPixelSize = 0;
                return false;
            }

            totalPixelSize += (unsigned long long)slicePitch;
            ++nimages;

            if (h > 1)
                h >>= 1;

            if (w > 1)
                w >>= 1;
        }
    }

    static_assert(sizeof(size_t) == 8, "Not a 64-bit platform!");

    nImages = nimages;
    pixelSize = static_cast<unsigned long long>(totalPixelSize);

    return true;
}
//-------------------------------------------------------------------------------------
// Fills in the image array entries
//-------------------------------------------------------------------------------------
bool SetupImageArray(
    unsigned char* pMemory,
    unsigned long long pixelSize,
    const TexMetadata& metadata,
    DWORD cpFlags,
    Image* images,
    unsigned long long nImages)
{
    assert(pMemory);
    assert(pixelSize > 0);
    assert(nImages > 0);

    if (!images)
        return false;
    unsigned long long index = 0;
    unsigned char* pixels = pMemory;
    const unsigned char* pEndBits = pMemory + pixelSize;

    if (metadata.arraySize == 0 || metadata.mipLevels == 0)
        return false;

    for (unsigned long long item = 0; item < metadata.arraySize; ++item)
    {
        unsigned long long w = metadata.width;
        unsigned long long h = metadata.height;

        for (unsigned long long level = 0; level < metadata.mipLevels; ++level)
        {
            if (index >= nImages)
                return false;

            unsigned long long rowPitch, slicePitch;
            if (FAILED(ComputePitch(metadata.format, w, h, rowPitch, slicePitch, cpFlags)))
                return false;

            images[index].width = w;
            images[index].height = h;
            images[index].format = metadata.format;
            images[index].rowPitch = rowPitch;
            images[index].slicePitch = slicePitch;
            images[index].pixels = pixels;
            ++index;

            pixels += slicePitch;
            if (pixels > pEndBits)
                return false;

            if (h > 1)
                h >>= 1;

            if (w > 1)
                w >>= 1;
        }
    }
    return true;
}

inline WICBitmapDitherType __cdecl GetWICDither(DWORD flags)
{
    static_assert(TEX_FILTER_DITHER == 0x10000, "TEX_FILTER_DITHER* flag values don't match mask");

    static_assert(static_cast<int>(TEX_FILTER_DITHER) == static_cast<int>(WIC_FLAGS_DITHER), "TEX_FILTER_DITHER* should match WIC_FLAGS_DITHER*");
    static_assert(static_cast<int>(TEX_FILTER_DITHER_DIFFUSION) == static_cast<int>(WIC_FLAGS_DITHER_DIFFUSION), "TEX_FILTER_DITHER* should match WIC_FLAGS_DITHER*");

    switch (flags & 0xF0000)
    {
    case TEX_FILTER_DITHER:
        return WICBitmapDitherTypeOrdered4x4;

    case TEX_FILTER_DITHER_DIFFUSION:
        return WICBitmapDitherTypeErrorDiffusion;

    default:
        return WICBitmapDitherTypeNone;
    }
}


DXGI_FORMAT GetUAVCompatableFormat(DXGI_FORMAT format)
{
    DXGI_FORMAT uavFormat = format;

    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        uavFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        break;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        uavFormat = DXGI_FORMAT_R32_FLOAT;
        break;
    }

    return uavFormat;
}




HRESULT WicLoadImageFromMemory(IWICImagingFactory* factory, void* iBuffer, DWORD bufferSize, TexMetadata* metadata, Image*& images, unsigned long long& nimages, unsigned char*& imageMemory, unsigned long long& memorySize, DWORD flags, DWORD cpFlags = CP_FLAGS_NONE)
{
    IWICStream* stream;
    HRESULT hr = factory->CreateStream(&stream);
    if (FAILED(hr))
        return hr;
    hr =stream->InitializeFromMemory((unsigned char*)iBuffer, bufferSize);
    if (FAILED(hr))
    {
        stream->Release();
        return hr;
    }
    
    IWICBitmapDecoder* decoder;
    //factory->CreateDecoderFromFilename(fileName, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder);
    hr = factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand,&decoder);
    if (FAILED(hr))
    {
        stream->Release();
        return hr;
    }

    IWICBitmapFrameDecode* frame;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr))
    {
        stream->Release();
        decoder->Release();
        return hr;
    }
    
    //Get metadata
    TexMetadata mdata = {};
    mdata.depth = 1;
    mdata.mipLevels = 1;
    //TODO: do i need these flags??
    mdata.miscFlags = 0;
    mdata.miscFlags2 = 0;
    //we just load 2d textures
    mdata.dimension = TEX_DIMENSION_TEXTURE2D;

    //DECODE METADATA----------------------------------------------------------------------------------------
    unsigned int w, h;
    hr = frame->GetSize(&w, &h);
    mdata.width = w;
    mdata.height = h;

    if (flags & WIC_FLAGS_ALL_FRAMES)
    {
        unsigned int fCount;
        hr = decoder->GetFrameCount(&fCount);
        if (FAILED(hr))
        {
            stream->Release();
            decoder->Release();
            frame->Release();
            return hr;
        }

        mdata.arraySize = fCount;
    }
    else
        mdata.arraySize = 1;

    //Determine format-------------------------------------------------------------------------------
    WICPixelFormatGUID pixelFormat;
    WICPixelFormatGUID convertGUID = {};
    hr = frame->GetPixelFormat(&pixelFormat);
    if (FAILED(hr))
    {
        stream->Release();
        decoder->Release();
        frame->Release();
        return hr;
    }
    DXGI_FORMAT format = WICToDXGI(pixelFormat);
    if (format == DXGI_FORMAT_UNKNOWN)
    {
        if (memcmp(&GUID_WICPixelFormat96bppRGBFixedPoint, &pixelFormat, sizeof(WICPixelFormatGUID)) == 0)
        {
            memcpy(&convertGUID, &GUID_WICPixelFormat96bppRGBFloat, sizeof(WICPixelFormatGUID));
            format = DXGI_FORMAT_R32G32B32_FLOAT;
        }
        else
        {
            for (size_t i = 0; i < _countof(g_WICConvert); ++i)
            {
                if (memcmp(&g_WICConvert[i].source, &pixelFormat, sizeof(WICPixelFormatGUID)) == 0)
                {
                    memcpy(&convertGUID, &g_WICConvert[i].target, sizeof(WICPixelFormatGUID));

                    format = WICToDXGI(g_WICConvert[i].target);
                    //assert(format != DXGI_FORMAT_UNKNOWN);
                    if (format == DXGI_FORMAT_UNKNOWN)
                    {
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
                    }
                    break;
                }
            }
        }

        //Handle special cases based on flags
        switch (format)
        {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
            if (flags & WIC_FLAGS_FORCE_RGB)
            {
                format = DXGI_FORMAT_R8G8B8A8_UNORM;
                memcpy(&convertGUID, &GUID_WICPixelFormat32bppRGBA, sizeof(WICPixelFormatGUID));
            }
            break;

        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
            if (flags & WIC_FLAGS_NO_X2_BIAS)
            {
                format = DXGI_FORMAT_R10G10B10A2_UNORM;
                memcpy(&convertGUID, &GUID_WICPixelFormat32bppRGBA1010102, sizeof(WICPixelFormatGUID));
            }
            break;

        case DXGI_FORMAT_B5G5R5A1_UNORM:
        case DXGI_FORMAT_B5G6R5_UNORM:
            if (flags & WIC_FLAGS_NO_16BPP)
            {
                format = DXGI_FORMAT_R8G8B8A8_UNORM;
                memcpy(&convertGUID, &GUID_WICPixelFormat32bppRGBA, sizeof(WICPixelFormatGUID));
            }
            break;
           
        case DXGI_FORMAT_R8_UNORM:
            if(!(flags & WIC_FLAGS_ALLOW_MONO))
            {
                format = DXGI_FORMAT_R8_UNORM;
                memcpy(&convertGUID, &GUID_WICPixelFormat8bppGray, sizeof(WICPixelFormatGUID));
            }
            break;

        default:
            break;
        }
    }
    if (format == DXGI_FORMAT_UNKNOWN)
    {
        stream->Release();
        decoder->Release();
        frame->Release();
        return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    }
    mdata.format = format;

    //make srgb
    if (!(flags & WIC_FLAGS_IGNORE_SRGB))
    {
        GUID containerFormat;
        hr = decoder->GetContainerFormat(&containerFormat);
        if (FAILED(hr))
            return hr;

        IWICMetadataQueryReader* metareader;
        hr = frame->GetMetadataQueryReader(&metareader);
        if (SUCCEEDED(hr))
        {
            bool sRGB = false;

            PROPVARIANT value;
            PropVariantInit(&value);

            if (memcmp(&containerFormat, &GUID_ContainerFormatPng, sizeof(GUID)) == 0)
            {
                if (SUCCEEDED(metareader->GetMetadataByName(L"/sRGB/RenderingIntent", &value)) && value.vt == VT_UI1)
                    sRGB = true;
            }
#if defined(_XBOX_ONE) && defined(_TITLE)
            else if (memcmp(&containerFormat, &GUID_ContainerFormatJpeg, sizeof(GUID)) == 0)
            {
                if (SUCCEEDED(metareader->GetMetadataByName(L"/app1/ifd/exif/{ushort=40961}", &value)) && value.vt == VT_UI2 && value.uiVal == 1)
                {
                    sRGB = true;
                }
            }
            else if (memcmp(&containerFormat, &GUID_ContainerFormatTiff, sizeof(GUID)) == 0)
            {
                if (SUCCEEDED(metareader->GetMetadataByName(L"/ifd/exif/{ushort=40961}", &value)) && value.vt == VT_UI2 && value.uiVal == 1)
                {
                    sRGB = true;
                }
            }
#else
            else if (SUCCEEDED(metareader->GetMetadataByName(L"System.Image.ColorSpace", &value)) && value.vt == VT_UI2 && value.uiVal == 1)
            {
                sRGB = true;
            }
#endif
            (void)PropVariantClearPtr(&value);

            if (sRGB)
                mdata.format = MakeSRGB(mdata.format);

            metareader->Release();
        }
        else if (hr == WINCODEC_ERR_UNSUPPORTEDOPERATION)
        {
            // Some formats just don't support metadata (BMP, ICO, etc.), so ignore this failure
            hr = S_OK;
        }

    }

    //convert the format if necessary and calculate its rowPitch, slicePitch and mipmaps. Finally retrieve the final buffer--------------------------------------------------
     //Initialize2D----------------------------------------------------------------------------------- (it is done in both decode single and multiple frames)
    if (!CalculateMipLevels(mdata.width, mdata.height, mdata.mipLevels))
    {
        stream->Release();
        decoder->Release();
        frame->Release();
        return HRESULT_FROM_WIN32(E_INVALIDARG);
    }
    unsigned long long pixelSize;
    if (!DetermineImageArray(mdata, cpFlags, nimages, pixelSize))
    {
        stream->Release();
        decoder->Release();
        frame->Release();
        return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
    }
    assert(images == nullptr);
    assert(imageMemory == nullptr);
    images = (Image*)HeapAlloc(GetProcessHeap(), 0, sizeof(Image) * nimages);
    //the heap alloc function alignment is defined by MEMORY_ALLOCATION_ALIGNEMT
    static_assert(MEMORY_ALLOCATION_ALIGNMENT == 16, "texture memory missaligned");
    imageMemory = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, pixelSize);
    memorySize = pixelSize;
    if (!SetupImageArray(imageMemory, pixelSize, mdata, cpFlags, images, nimages))
    {
        HeapFree(GetProcessHeap(), 0, images);
        images = nullptr;
        HeapFree(GetProcessHeap(), 0, imageMemory);
        imageMemory = nullptr;
        stream->Release();
        decoder->Release();
        frame->Release();
        return HRESULT_FROM_WIN32(E_FAIL);
    }
    //----------------------------------------------------------------------------------------------
    if ((mdata.arraySize > 1) && (flags & WIC_FLAGS_ALL_FRAMES))
    {
        //hr = DecodeMultiframe(flags, mdata, decoder.Get(), image);
        WICPixelFormatGUID sourceGUID;
        if (!DXGIToWIC(mdata.format, sourceGUID))
        {
            HeapFree(GetProcessHeap(), 0, images);
            images = nullptr;
            HeapFree(GetProcessHeap(), 0, imageMemory);
            imageMemory = nullptr;
            stream->Release();
            decoder->Release();
            frame->Release();
            return HRESULT_FROM_WIN32(E_FAIL);
        }
        for (unsigned long long index = 0; index < mdata.arraySize; ++index)
        {
            const Image* img = &images[index * (mdata.mipLevels)];//&images[index * (mdata.mipLevels) + 0];//images[index * (m_metadata.mipLevels) + mip]; (si la taxture es 1d o 2d (en 3d esta mal))
            if (img->rowPitch > UINT32_MAX || img->slicePitch > UINT32_MAX)
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return HRESULT_FROM_WIN32(E_FAIL);
            }
            frame->Release();
            hr =decoder->GetFrame(static_cast<unsigned int>(index), &frame);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return hr;
            }
            WICPixelFormatGUID pfGuid;
            hr = frame->GetPixelFormat(&pfGuid);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return hr;
            }
            hr = frame->GetSize(&w, &h);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return hr;
            }
            if (w == mdata.width && h == mdata.height)
            {
                // This frame does not need resized
                if (memcmp(&pfGuid, &sourceGUID, sizeof(WICPixelFormatGUID)) == 0)
                {
                    hr = frame->CopyPixels(nullptr, static_cast<UINT>(img->rowPitch), static_cast<UINT>(img->slicePitch), img->pixels);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        return hr;
                    }
                }
                else
                {
                    IWICFormatConverter* FC;
                    hr = factory->CreateFormatConverter(&FC);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        return hr;
                    }

                    BOOL canConvert = FALSE;
                    hr = FC->CanConvert(pfGuid, sourceGUID, &canConvert);
                    if (FAILED(hr) || !canConvert)
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        FC->Release();
                        return E_UNEXPECTED;
                    }

                    hr = FC->Initialize(frame, sourceGUID, GetWICDither(flags), nullptr, 0, WICBitmapPaletteTypeMedianCut);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        FC->Release();
                        return hr;
                    }

                    hr = FC->CopyPixels(nullptr, static_cast<UINT>(img->rowPitch), static_cast<UINT>(img->slicePitch), img->pixels);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        FC->Release();
                        return hr;
                    }
                    FC->Release();
                }
            }
            else
            {
                // This frame needs resizing
                IWICBitmapScaler* scaler;
                hr = factory->CreateBitmapScaler(&scaler);
                if (FAILED(hr))
                {
                    HeapFree(GetProcessHeap(), 0, images);
                    images = nullptr;
                    HeapFree(GetProcessHeap(), 0, imageMemory);
                    imageMemory = nullptr;
                    stream->Release();
                    decoder->Release();
                    frame->Release();
                    return hr;
                }
                hr = scaler->Initialize(frame, static_cast<UINT>(mdata.width), static_cast<UINT>(mdata.height), GetWICInterp(flags));
                if (FAILED(hr))
                {
                    HeapFree(GetProcessHeap(), 0, images);
                    images = nullptr;
                    HeapFree(GetProcessHeap(), 0, imageMemory);
                    imageMemory = nullptr;
                    stream->Release();
                    decoder->Release();
                    frame->Release();
                    scaler->Release();
                    return hr;
                }
                WICPixelFormatGUID pfScaler;
                hr = scaler->GetPixelFormat(&pfScaler);
                if (FAILED(hr))
                {
                    HeapFree(GetProcessHeap(), 0, images);
                    images = nullptr;
                    HeapFree(GetProcessHeap(), 0, imageMemory);
                    imageMemory = nullptr;
                    stream->Release();
                    decoder->Release();
                    frame->Release();
                    scaler->Release();
                    return hr;
                }
                if (memcmp(&pfScaler, &sourceGUID, sizeof(WICPixelFormatGUID)) == 0)
                {
                    hr = scaler->CopyPixels(nullptr, static_cast<UINT>(img->rowPitch), static_cast<UINT>(img->slicePitch), img->pixels);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        scaler->Release();
                        return hr;
                    }
                }
                else
                {
                    // The WIC bitmap scaler is free to return a different pixel format than the source image, so here we
                    // convert it to our desired format
                    IWICFormatConverter* FC;
                    hr = factory->CreateFormatConverter(&FC);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        scaler->Release();
                        return hr;
                    }

                    BOOL canConvert = FALSE;
                    hr = FC->CanConvert(pfScaler, sourceGUID, &canConvert);
                    if (FAILED(hr) || !canConvert)
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        scaler->Release();
                        FC->Release();
                        return E_UNEXPECTED;
                    }

                    hr = FC->Initialize(scaler, sourceGUID, GetWICDither(flags), nullptr, 0, WICBitmapPaletteTypeMedianCut);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        scaler->Release();
                        FC->Release();
                        return hr;
                    }

                    hr = FC->CopyPixels(nullptr, static_cast<UINT>(img->rowPitch), static_cast<UINT>(img->slicePitch), img->pixels);
                    if (FAILED(hr))
                    {
                        HeapFree(GetProcessHeap(), 0, images);
                        images = nullptr;
                        HeapFree(GetProcessHeap(), 0, imageMemory);
                        imageMemory = nullptr;
                        stream->Release();
                        decoder->Release();
                        frame->Release();
                        scaler->Release();
                        FC->Release();
                        return hr;
                    }

                    FC->Release();
                }
                scaler->Release();
            }
        }
    }
    else
    {
        //hr = DecodeSingleFrame(flags, mdata, convertGUID, frame.Get(), image);
        if (images[0].rowPitch > UINT32_MAX || images[0].slicePitch > UINT32_MAX)
        {
            HeapFree(GetProcessHeap(), 0, images);
            images = nullptr;
            HeapFree(GetProcessHeap(), 0, imageMemory);
            imageMemory = nullptr;
            stream->Release();
            decoder->Release();
            frame->Release();
            return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
        }
        if (memcmp(&convertGUID, &MY_GUID_NULL, sizeof(GUID)) == 0)
        {
            hr = frame->CopyPixels(nullptr, static_cast<unsigned int>(images->rowPitch), static_cast<unsigned int>(images->slicePitch), images->pixels);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return hr;
            }
        }
        else //we need to convert
        {
            IWICFormatConverter* FC;
            hr = factory->CreateFormatConverter(&FC);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                return hr;
            }

            //WICPixelFormatGUID pixelFormat;
            //hr = frame->GetPixelFormat(&pixelFormat);
            BOOL canConvert = FALSE;
            hr = FC->CanConvert(pixelFormat, convertGUID, &canConvert);
            if (FAILED(hr) || !canConvert)
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                FC->Release();
                return HRESULT_FROM_WIN32(E_UNEXPECTED);
            }

            hr = FC->Initialize(frame, convertGUID, GetWICDither(flags), nullptr, 0, WICBitmapPaletteTypeMedianCut);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                FC->Release();
                return hr;
            }

            hr = FC->CopyPixels(nullptr, static_cast<unsigned int>(images->rowPitch), static_cast<unsigned int>(images->slicePitch), images->pixels);
            if (FAILED(hr))
            {
                HeapFree(GetProcessHeap(), 0, images);
                images = nullptr;
                HeapFree(GetProcessHeap(), 0, imageMemory);
                imageMemory = nullptr;
                stream->Release();
                decoder->Release();
                frame->Release();
                FC->Release();
                return hr;
            }
            FC->Release();
        }
    }


    if (metadata)
        memcpy(metadata, &mdata, sizeof(TexMetadata));

    stream->Release();
    decoder->Release();
    frame->Release();
    return S_OK;
    
}

//DIRECXTexture (with some modifications)-------------------------------------------------------------------------------------------------------------------