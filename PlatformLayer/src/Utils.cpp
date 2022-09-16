namespace Containers
{
	template <typename T, unsigned int incSize = 5>
	class MyQueue
	{
	public:
		MyQueue() : ptr(nullptr), size(0), capacity(incSize) { ptr = (T*)HeapAlloc(GetProcessHeap(), 0, capacity * sizeof(T)); }
		MyQueue(const MyQueue& queue) : ptr(nullptr), size(queue.size), capacity(queue.capacity)
		{
			ptr = (T*)HeapAlloc(GetProcessHeap(), 0, capacity * sizeof(T));
			memcpy(ptr, queue.ptr, size);
		}
		MyQueue(MyQueue&& queue) : ptr(queue.ptr), size(queue.size), capacity(incSize) { queue.ptr = nullptr; queue.size = 0; }
		~MyQueue() { HeapFree(GetProcessHeap(), 0, ptr); }
		bool Empty() const { return size == 0; }
		unsigned int Size() const { return size; }
		T Front() const { return ptr[0]; }
		T Back() const { return ptr[size - 1]; }
		void Push(const T& value)
		{
			++size;
			if (size > capacity)
			{
				capacity += incSize;
				T* ptr2 = (T*)HeapAlloc(GetProcessHeap(), 0, capacity * sizeof(T));
				memcpy(ptr2, ptr, ((capacity - incSize) * sizeof(T)));
				HeapFree(GetProcessHeap(), 0, ptr);
				ptr = ptr2;
			}
			ptr[size - 1] = value;
		}
		void Pop()
		{
			for (int i = 0; i < size - 1; ++i)
			{
				ptr[i] = ptr[i + 1];
			}
			--size;
		}
	private:
		T* ptr;
		//num of elements
		unsigned int size;
		//max num of elements
		unsigned int capacity;
	};
}

unsigned int StrLen(char* str)
{
	unsigned int size = 0;
	while (str[size] != '\0')
		++size;
	return size;
}

int StrCmp(const char* str1, const char* str2)
{
	unsigned int i = 0;
	while (str1[i] != '\0')
	{
		if (str1[i] != str2[i])
		{
			if (str1[i] < str2[i])
				return 1;
			else
				return -1;
		}
		++i;
	}
	if (str1[i] == str2[i])
		return 0;
	else if (str1[i] < str2[i])
		return 1;
	else
		return -1;
}

template <class _Ty>
constexpr _Ty Max(_Ty left, _Ty right)
{
	return (left < right) ? right : left;
}

char* IntegerToChar(int in)
{
	char stackChars[12] = {};
	int i = 10;
	if (in == 0) {
		i = 9;
		stackChars[10] = '0';
	}
	bool negative = in & (1 << 31);
	if (negative)
		in = -in;
	for (; in != 0; in /= 10, --i)
	{
		stackChars[i] = ((char)(in % 10)) + '0';
	}
	negative ? stackChars[i] = (char)45 : ++i;
	unsigned short size = 12 - i;
	char* ret = (char*)HeapAlloc(GetProcessHeap(), 0, size);
	memcpy(ret, stackChars + i, size);
	return ret;
}

char* ConvertFloatToChar(float inFloat)
{
	int integralPart = (int)inFloat;
	float decimal = inFloat - (float)integralPart;
	int intDecimal = (int)(decimal * 1000000000);
	char* integerChar = IntegerToChar(integralPart);
	char* decimalChar = IntegerToChar(intDecimal);
	unsigned int iSize = StrLen(integerChar);
	unsigned int dSize = StrLen(decimalChar);
	char* result = (char*)HeapAlloc(GetProcessHeap(), 0, iSize + dSize + 2);
	memcpy(result, integerChar, iSize + 1);
	memcpy(result + iSize + 1, decimalChar, dSize + 1);
	result[iSize] = '.';
	HeapFree(GetProcessHeap(), 0, integerChar);
	HeapFree(GetProcessHeap(), 0, decimalChar);
	return result;
}