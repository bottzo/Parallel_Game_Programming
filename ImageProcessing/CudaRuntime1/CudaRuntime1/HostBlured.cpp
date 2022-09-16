//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image/stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image/stb_image_write.h"
//
//#include <windows.h>
//
//int main(int argc, char* argv[])
//{
//	if (argc == 4 && strcmp(argv[1], "bl") == 0)
//	{
//		int width, height, channels;
//		unsigned char* img = stbi_load(argv[2], &width, &height, &channels, 0);
//		if (img == 0)
//		{
//			printf("Error loading the image from disk");
//			return 0;
//		}
//		size_t img_size = width * height * channels;
//		int gray_channels = channels == 4 ? 2 : 1;
//		int gray_img_size = width * height * gray_channels;
//		unsigned char* gray_img = (unsigned char*)malloc(gray_img_size);
//		LARGE_INTEGER freq;
//		LARGE_INTEGER first_count;
//		LARGE_INTEGER second_count;
//		QueryPerformanceFrequency(&freq);
//		QueryPerformanceCounter(&first_count);
//		//1. separar els channels
//
//		//2. for Per cada channel
//		for (unsigned char c = 0; c < channels; ++c)
//		{
//			//3.for per cada pixel del channel
//			for ()
//			{
//				//4. calcular el kernel per cada un
//			}
//		}
//		for (unsigned char* p = img, *pg = gray_img; p != img + img_size; p+= channels, pg += gray_channels) 
//		{
//			//greyPixels[index] = 0.299f * colorPixels[index].x + 0.587f * colorPixels[index].y + 0.114f * colorPixels[index].z;
//			*pg = (unsigned char)(((*p) * 0.299f + (*(p + 1)) * 0.587f + (*(p + 2)) * 0.114f));
//			if (channels == 4)
//				 *(p + 3) = *(pg + 1);
//		}
//		QueryPerformanceCounter(&second_count);
//		long long counts = second_count.QuadPart - first_count.QuadPart;
//		double ms = 1000 * ((double)counts / (double)freq.QuadPart);
//		printf("\n%f ms \n", ms);
//		stbi_write_jpg(argv[2], width, height, gray_channels, gray_img, 100);
//		stbi_image_free(img);
//		free(gray_img);
//	}
//	else 
//	{
//		printf("First parameter is the source and second parameter is the destination");
//	}
//	return 0;
//}