#include <dxgi1_6.h>
typedef HRESULT(WINAPI* CREATE_DXGI_FACTORY1)(REFIID riid, _COM_Outptr_ void** ppFactory);
typedef HRESULT(WINAPI* CREATE_DXGI_FACTORY2)(UINT Flags, REFIID riid, _COM_Outptr_ void** ppFactory);
CREATE_DXGI_FACTORY1 CreateDXGIFactory1Ptr = nullptr;
CREATE_DXGI_FACTORY2 CreateDXGIFactory2Ptr = nullptr;
#include "d3d12.h"
//Dynamic library loading function pointers
#ifdef _DEBUG
PFN_D3D12_GET_DEBUG_INTERFACE D3D12GetDebugInterfacePtr = nullptr;
#endif // _DEBUG
PFN_D3D12_CREATE_DEVICE D3D12CreateDevicePtr = nullptr;
PFN_D3D12_SERIALIZE_ROOT_SIGNATURE D3D12SerializeRootSignaturePtr = nullptr;
PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE D3D12SerializeVersionedRootSignaturePtr = nullptr;
#include "d3dx12.h"

//always include before windows.h
//the only function that needs the windows.h inclusion (before) is DirectX::XMVerifyCPUSupport();
#include <DirectXMath.h>

//To use the ComPtr
//#include <wrl.h>
//using namespace Microsoft::WRL;

//exports to set the agility sdk parameters to let the d3d12.dll load the correct D3D12Core.dll (system or apps)
extern "C" 
{ 
	__declspec(dllexport) extern const UINT D3D12SDKVersion = 600; 
	__declspec(dllexport) extern const char* D3D12SDKPath = u8".\\D3D12\\"; 
}


typedef struct Fence {
	ID3D12Fence* iFence;
	unsigned long long value;
	HANDLE event;
}Fence;

typedef struct syncAllocator
{
	ID3D12CommandAllocator* cAllocator;
	unsigned long long fenceValue;
} syncAllocator;

typedef struct GraphicsCommandQueue {
	ID3D12CommandQueue* iCQueue;
	Fence fence;
	Containers::MyQueue<ID3D12GraphicsCommandList2*> cLists;
	Containers::MyQueue<syncAllocator> cAllocators;
}GraphicsCommandQueue;

void FlushComandQueue(GraphicsCommandQueue& cQueue)
{
	cQueue.iCQueue->Signal(cQueue.fence.iFence, ++cQueue.fence.value);
	if (cQueue.fence.iFence->GetCompletedValue() >= cQueue.fence.value) {
		cQueue.fence.iFence->SetEventOnCompletion(cQueue.fence.value, cQueue.fence.event);
		WaitForSingleObject(cQueue.fence.event, INFINITE);
	}
}


typedef struct DirectXRenderStruct {
	//IDXGI
	IDXGIAdapter4* adapter;
	IDXGISwapChain4* sChain;
	//ID3D12
	ID3D12Device2* device; //TODO: there is just 1 device for the moment (TODO: how to use multiple devices if aviable)
	GraphicsCommandQueue cQueue;
	//Resources
	ID3D12Resource* depthBuffer;
	ID3D12DescriptorHeap* dsvDescHeap;
	ID3D12Resource** backBuffers;
	ID3D12DescriptorHeap* rtvDescHeap;
	unsigned int backBufferCount;
	unsigned int currentBackBufferIndex;
	unsigned long long rtvFenceValues[3];

	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	bool tearingSupport;
	bool vSync;

}DirectXRenderStruct;

void InitDebugLayer()
{
	//Enable it before using anithing related to D3D12 or DXGI
#ifdef _DEBUG
	ID3D12Debug* debugController;
	//PFN_D3D12_GET_DEBUG_INTERFACE (to query at runtime with the GetProcAddress)
	if (SUCCEEDED(D3D12GetDebugInterfacePtr(IID_PPV_ARGS(&debugController))))
	{
		debugController->EnableDebugLayer();
		debugController->Release();
	}
#endif // _DEBUG

}

//An adapter is a representation of a video card
IDXGIAdapter4* GetAdapter(bool useWarp)
{
	IDXGIFactory4* dxgiFactory;
	unsigned int dxgiFactoryFlags = 0;
#ifdef _DEBUG
	dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
	CreateDXGIFactory2Ptr(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory));

	IDXGIAdapter1* dxgiAdapter1 = nullptr;
	IDXGIAdapter4* dxgiAdapter4 = nullptr;

	if (useWarp)
	{
		dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter1));
		dxgiAdapter1->QueryInterface(IID_PPV_ARGS(&dxgiAdapter4));
		dxgiAdapter1->Release();
	}
	else
	{
		SIZE_T maxDedicatedVideoMemory = 0;
		for(unsigned int i = 0; dxgiFactory->EnumAdapters1(i,&dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; ++i)
		{
			DXGI_ADAPTER_DESC1 dxgiAdapterDesc1;
			dxgiAdapter1->GetDesc1(&dxgiAdapterDesc1);

			// Check to see if the adapter can create a D3D12 device without actually 
			// creating it. The adapter with the largest dedicated video memory
			// is favored.
			if ((dxgiAdapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
				SUCCEEDED(D3D12CreateDevicePtr(dxgiAdapter1,D3D_FEATURE_LEVEL_12_1, __uuidof(ID3D12Device), nullptr)) &&
				dxgiAdapterDesc1.DedicatedVideoMemory > maxDedicatedVideoMemory)
			{
				maxDedicatedVideoMemory = dxgiAdapterDesc1.DedicatedVideoMemory;
				//TODO://ThrowIfFailed(dxgiAdapter1.As(&dxgiAdapter4));
				if(dxgiAdapter4!=nullptr)
					dxgiAdapter4->Release();
				dxgiAdapter1->QueryInterface(IID_PPV_ARGS(&dxgiAdapter4));
			}
			dxgiAdapter1->Release();
			dxgiAdapter1 = nullptr;
		}
	}

	dxgiFactory->Release();
	//dxgiAdapter4->Release();
	return dxgiAdapter4;
}

ID3D12Device2* CreateDevice(IDXGIAdapter4* adapter)
{
	//Creating a device
	ID3D12Device2* device2 = nullptr;
	//if the adapter was null we would be using the first adapter returned by IDXGIFactory1::EnumAdapters
	///*if (SUCCEEDED(*/D3D12CreateDevicePtr(adapter, D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device2));
	/*if (SUCCEEDED(*/D3D12CreateDevicePtr(adapter, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device2));

#ifdef _DEBUG
	// Enable debug messages in debug mode.
	ID3D12InfoQueue* infoQueue;
	if(device2 != nullptr && device2->QueryInterface(IID_PPV_ARGS(&infoQueue)) != E_NOINTERFACE)
	{
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);

		//To ignore or supress some warnings or messages
		// Suppress whole categories of messages
		//D3D12_MESSAGE_CATEGORY Categories[] = {};
		
		// Suppress messages based on their severity level
		D3D12_MESSAGE_SEVERITY Severities[] =
		{
			D3D12_MESSAGE_SEVERITY_INFO
		};

		// Suppress individual messages by their ID
		D3D12_MESSAGE_ID DenyIds[] = {
			D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,   // I'm really not sure how to avoid this message.
			D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                         // This warning occurs when using capture frame while graphics debugging.
			D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                       // This warning occurs when using capture frame while graphics debugging.
			//D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED,                  // This warning occurs when creating a commited resource using a default heap and initial state type D3D12_RESOURCE_STATE_COPY_DEST != than D3D12_RESOURCE_STATE_COMMON 
		};
		D3D12_INFO_QUEUE_FILTER NewFilter = {};
		//NewFilter.DenyList.NumCategories = _countof(Categories);
		//NewFilter.DenyList.pCategoryList = Categories;
		NewFilter.DenyList.NumSeverities = _countof(Severities);
		NewFilter.DenyList.pSeverityList = Severities;
		NewFilter.DenyList.NumIDs = _countof(DenyIds);
		NewFilter.DenyList.pIDList = DenyIds;
		//TODO: ThrowIfFailed(infoQueue->PushStorageFilter(&NewFilter));
		infoQueue->PushStorageFilter(&NewFilter);

		infoQueue->Release();
	}
#endif

	//TODO: adapter->Release();
	return device2;
}

//The CommandQueues executes CommandLists (one fence value per CommandQueue)
ID3D12CommandQueue* CreateCommandQueue(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
{
	ID3D12CommandQueue* cQueue = nullptr;

	D3D12_COMMAND_QUEUE_DESC desc = {};
	//D3D12_COMMAND_LIST_TYPE_DIRECT: draw + compute + copy commands
	//D3D12_COMMAND_LIST_TYPE_COMPUTE: compute + copy commands
	//D3D12_COMMAND_LIST_TYPE_COPY: copy commands
	desc.Type = type;
	desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	//if we would use more than 1 GPU set different to 0 (Multi-adapter)
	desc.NodeMask = 0;

	device->CreateCommandQueue(&desc, IID_PPV_ARGS(&cQueue));

	return cQueue;
}

//Variable refresh rate displays (NVidia’s G-Sync and AMD’s FreeSync) 
//require tearing to be enabled in the DirectX 12 application to function correctly
bool CheckTearingSupport()
{
	bool allowTearing = false;

	// Rather than create the DXGI 1.5 factory interface directly, we create the
	// DXGI 1.4 interface and query for the 1.5 interface. This is to enable the 
	// graphics debugging tools which will not support the 1.5 factory interface 
	// until a future update.
	IDXGIFactory4* factory4;
	if (SUCCEEDED(CreateDXGIFactory1Ptr(IID_PPV_ARGS(&factory4))))
	{
		IDXGIFactory5* factory5;
		if (factory4->QueryInterface(IID_PPV_ARGS(&factory5)) != E_NOINTERFACE)
			if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,&allowTearing, sizeof(allowTearing))))
				allowTearing = false;

		factory5->Release();
		factory4->Release();
	}

	return allowTearing;
}

//The primary purpose of the swap chain is to present the rendered image to the screen.
//The swap chain stores no less than two buffers that are used to render the scene.
//The buffer that is currently being rendered to is called the back buffer and
//the buffer that is currently being presented is called the front buffer.
IDXGISwapChain4* CreateSwapChain(HWND hWnd, ID3D12CommandQueue* cQueue, unsigned int width, unsigned int height, unsigned int bufferCount)
{
	IDXGISwapChain4* dxgiSwapChain4 = nullptr;
	IDXGIFactory4* dxgiFactory4;
	unsigned int createFactoryFlags = 0;

#ifdef _DEBUG
	createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

	CreateDXGIFactory2Ptr(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory4));

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	//if width and height are 0 they will be set to the ones of the window when swap chain created
	//we can always query them with IDXGISwapChain1::GetDesc1()
	swapChainDesc.Width = width;
	swapChainDesc.Height = height;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.Stereo = false;
	swapChainDesc.SampleDesc = { 1, 0 };
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.BufferCount = bufferCount;
	swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	// It is recommended to always allow tearing if tearing support is available.
	swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	IDXGISwapChain1* swapChain1;
	//TODO: throw if failed on the next 3 funcs
	dxgiFactory4->CreateSwapChainForHwnd(cQueue, hWnd, &swapChainDesc, nullptr, nullptr, &swapChain1);

	// Disable the Alt+Enter fullscreen toggle feature. Switching to fullscreen
	// will be handled manually.
	dxgiFactory4->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER);

	//swapChain1.As(&dxgiSwapChain4));
	swapChain1->QueryInterface(IID_PPV_ARGS(&dxgiSwapChain4));
	
	dxgiFactory4->Release();
	swapChain1->Release();

	return dxgiSwapChain4;
}


ID3D12DescriptorHeap* CreateRTVDescriptorHeap(ID3D12Device* device, unsigned int numFrames, IDXGISwapChain4* swapChain, ID3D12Resource** backBuffers, unsigned int& currentBackBufferIndex)
{
	ID3D12DescriptorHeap* dHeap;
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = numFrames;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&dHeap));

	unsigned int incSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(dHeap->GetCPUDescriptorHandleForHeapStart());
	for (unsigned int i = 0; i < numFrames; ++i)
	{
		ID3D12Resource* backBuffer;
		//TODO: throw on fail
		swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffer));
		device->CreateRenderTargetView(backBuffer, nullptr, rtvHandle);
		backBuffers[i] = backBuffer;
		rtvHandle.Offset(incSize);
	}

	currentBackBufferIndex = swapChain->GetCurrentBackBufferIndex();
	return dHeap;
}

// Update the render target views for the swapchain back buffers.
void UpdateRenderTargetViews(ID3D12Device* device, ID3D12DescriptorHeap* rtvDHeap, unsigned int bufferCount, IDXGISwapChain* swapChain, ID3D12Resource** backBuffers)
{

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDHeap->GetCPUDescriptorHandleForHeapStart());
	unsigned int incSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	for (unsigned int i = 0; i < bufferCount; ++i)
	{
		ID3D12Resource* backBuffer;
		swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffer));

		device->CreateRenderTargetView(backBuffer, nullptr, rtvHandle);

		backBuffers[i] = backBuffer;

		rtvHandle.Offset(incSize);
	}
}


ID3D12CommandAllocator* CreateCommandAllocator(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
{
	ID3D12CommandAllocator* allocator = nullptr;
	//TODO: throw if failed
	device->CreateCommandAllocator(type,IID_PPV_ARGS(&allocator));

	return allocator;
}

ID3D12GraphicsCommandList1* CreateGraphicsCommandList(ID3D12Device* device, ID3D12CommandAllocator* allocator, ID3D12PipelineState* pipelineState, D3D12_COMMAND_LIST_TYPE type)
{
	ID3D12GraphicsCommandList1* cList = nullptr;
	device->CreateCommandList(0,type,allocator, pipelineState,IID_PPV_ARGS(&cList));
	//TODO: close the list just created???
	//cList->Close();
	return cList;
}

ID3D12RootSignature* CreateRootSignature(ID3D12Device* device)
{
	ID3D12RootSignature* rootSignature = nullptr;
	ID3DBlob* rSBlob;
	ID3DBlob* errorBlob = nullptr;

	// first we check if we support highest root signature version 
	D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};
	featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
	if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;

	D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS; //|
		//D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;


	CD3DX12_ROOT_PARAMETER1 rootParameters[2];
	rootParameters[0].InitAsConstants(/*sizeof(XMMATRIX) / 4*/16, 0, 0, D3D12_SHADER_VISIBILITY_VERTEX);
	//using the raw descriptor (1 indirection to acces but 2 of space in root sig)
	//rootParameters[1].InitAsShaderResourceView(0,0,D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_PIXEL);
	//if i want to use a descriptor table (2 indirection to acces but 1 of space in root sig)
	CD3DX12_DESCRIPTOR_RANGE1 descriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
	rootParameters[1].InitAsDescriptorTable(1, &descriptorRange, D3D12_SHADER_VISIBILITY_PIXEL);

	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDescription;
	CD3DX12_STATIC_SAMPLER_DESC linearRepeatSampler(0, D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR);
	//CD3DX12_STATIC_SAMPLER_DESC anisotropicSampler(0, D3D12_FILTER_ANISOTROPIC);
	//D3D12_STATIC_SAMPLER_DESC staticSampler = {};
	//staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	//staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	//staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	//staticSampler.ShaderRegister = 0;
	//staticSampler.RegisterSpace = 0;
	rootSignatureDescription.Init_1_1(_countof(rootParameters), rootParameters, 1, &linearRepeatSampler, rootSignatureFlags);

	D3DX12SerializeVersionedRootSignature(&rootSignatureDescription,featureData.HighestVersion, &rSBlob, &errorBlob);
	device->CreateRootSignature(0, rSBlob->GetBufferPointer(),rSBlob->GetBufferSize(), IID_PPV_ARGS(&rootSignature));

	rSBlob->Release();
	if(errorBlob)
		errorBlob->Release();

	return rootSignature;
}

//if we don't upload the buffer resource using an intermediate upload buffer we would have to use an upload heap to send and render the data
//we are trying to copy the vertex and index buffer from CPU to GPU memory
//ID3D12Resource* vertexBuffer;
//The heap type upload is suited for CPU-write-once GPU-read-once data. differing from type_default heaps that provides more GPU mem bandwith this type provides CPU access (map//unmap)
//CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
//CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer();
//creates both the resource and the heap to hold it
//device->CreateCommittedResource(vertexBuffer);
//Get a CPU pointer to the resource
//vertexBuffer->Map();
//Copy the data of the vertex shader to the GPU
//memcpy();
// invalidate the CPU pointer to the resource
//vertexBuffer->Unmap();
void UpdateBufferResource(ID3D12Device2* device, ID3D12GraphicsCommandList1* cList, ID3D12Resource** destinationResource, ID3D12Resource** intermediateResource, const void* bufferData, size_t numElements, size_t elementSize, D3D12_RESOURCE_FLAGS flags)
{
	size_t bufferSize = numElements * elementSize;

	auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, flags);

	device->CreateCommittedResource(
		&heapProperties,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		//TODO: why can't i create the resource in the state copy dest state?? // does it matter??
		D3D12_RESOURCE_STATE_COMMON,//D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(destinationResource));

	// Create an committed resource for the upload.
	if (bufferData)
	{
		heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

		device->CreateCommittedResource(
			&heapProperties,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(intermediateResource));

		D3D12_SUBRESOURCE_DATA subresourceData = {};
		subresourceData.pData = bufferData;
		subresourceData.RowPitch = bufferSize;
		subresourceData.SlicePitch = subresourceData.RowPitch;

		//the UpdateSubresources function is used to upload the CPU buffer data to the GPU resource in a default heap
		//using an intermediate buffer in an upload heap.
		UpdateSubresources(cList,
			*destinationResource, *intermediateResource,
			0, 0, 1, &subresourceData);
	}
}

// Transition a resource
void TransitionResource(ID3D12GraphicsCommandList1* commandList, ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState)
{
	CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, beforeState, afterState);

	commandList->ResourceBarrier(1, &barrier);
}

void ResizeDepthBuffer(ID3D12Device* device, ID3D12Resource** depthBuffer, ID3D12DescriptorHeap* dsvHeap, int width, int height, bool contentLoaded, GraphicsCommandQueue& cQueue)
{
	if (contentLoaded)
	{
		// Flush any GPU commands that might be referencing the depth buffer.
		//TODO: flush required but unecesary for now because the function that calls this already flushes
		//Maybe eliminate this funcion and jus call this on the global onResize
		//FLUSH 
		FlushComandQueue(cQueue);

		width = width > 1 ? width : 1;
		height = height > 1 ? height : 1;

		// Resize screen dependent resources.
		// Create a depth buffer.
		D3D12_CLEAR_VALUE optimizedClearValue = {};
		optimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		optimizedClearValue.DepthStencil = { 1.0f, 0 };

		//TODO: en el tutorial ell no necessita fer release !!!
		if ((*depthBuffer) != nullptr)
			(*depthBuffer)->Release();
			CD3DX12_HEAP_PROPERTIES prop(D3D12_HEAP_TYPE_DEFAULT);
			CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
			device->CreateCommittedResource(
				&prop,
				D3D12_HEAP_FLAG_NONE,
				&desc,
				D3D12_RESOURCE_STATE_DEPTH_WRITE,
				&optimizedClearValue,
				IID_PPV_ARGS(depthBuffer)
			);
		
		// Update the depth-stencil view.
		D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
		dsv.Format = DXGI_FORMAT_D32_FLOAT;
		dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		dsv.Texture2D.MipSlice = 0;
		dsv.Flags = D3D12_DSV_FLAG_NONE;

		device->CreateDepthStencilView(*depthBuffer, &dsv,
			dsvHeap->GetCPUDescriptorHandleForHeapStart());
	}
}


//TODO: call these on the windowproc func
void OnResize(int width, int height, bool contentLoaded, DirectXRenderStruct& renderer)
{
	//if(width != windowClientRect.right|| height != windowClientRect.bottom)
	{
		if (contentLoaded) {
			//FLUSH
			FlushComandQueue(renderer.cQueue);

			for (unsigned int i = 0; i < renderer.backBufferCount; ++i)
			{
				renderer.backBuffers[i]->Release();
			}
			
			DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
			renderer.sChain->GetDesc(&swapChainDesc);
			renderer.sChain->ResizeBuffers(renderer.backBufferCount, width, height, swapChainDesc.BufferDesc.Format, swapChainDesc.Flags);
			renderer.currentBackBufferIndex = renderer.sChain->GetCurrentBackBufferIndex();
			UpdateRenderTargetViews(renderer.device, renderer.rtvDescHeap, renderer.backBufferCount, renderer.sChain, renderer.backBuffers);
		}
		
		renderer.viewport = CD3DX12_VIEWPORT(0.0f, 0.0f,(float)width, (float)height);
		//ResizeDepthBuffer(device, depthBuffer, dsvHeap, width, height, contentLoaded);
		ResizeDepthBuffer(renderer.device, &renderer.depthBuffer, renderer.dsvDescHeap, width, height, contentLoaded, renderer.cQueue);
	}
}

DirectX::XMMATRIX modelMatrix;
DirectX::XMMATRIX viewMatrix;
DirectX::XMMATRIX projectionMatrix;
double angle = 0;
void OnUpdate(float dt)
{
	PIXScopedEvent(PIX_COLOR_INDEX(0), "Update");
	//TODO: chck for cpu avx support
	bool CPUSupportsInstructionSet = DirectX::XMVerifyCPUSupport();
	 
	
	// Update the model matrix.
	//float angle = static_cast<float>(e.TotalTime * 90.0);
	angle += dt * 150;
	const DirectX::XMVECTOR rotationAxis = DirectX::XMVectorSet(1, 0, 0, 0);
	modelMatrix = DirectX::XMMatrixRotationAxis(rotationAxis, DirectX::XMConvertToRadians(angle));
	
	// Update the view matrix.
	const DirectX::XMVECTOR eyePosition = DirectX::XMVectorSet(0, 0, -10, 1);
	const DirectX::XMVECTOR focusPoint =  DirectX::XMVectorSet(0, 0, 0, 1);
	const DirectX::XMVECTOR upDirection = DirectX::XMVectorSet(0, 1, 0, 0);
	viewMatrix = DirectX::XMMatrixLookAtLH(eyePosition, focusPoint, upDirection);
	
	// Update the projection matrix.
	float aspectRatio = windowClientRect.right / static_cast<float>(windowClientRect.bottom);
	float fov = 60;
	projectionMatrix = DirectX::XMMatrixPerspectiveFovLH(DirectX::XMConvertToRadians(fov), aspectRatio, 0.1f, 100.0f);

}

float vertices[48] = {
	-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // 0
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // 1
	 1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 0.0f, // 2
	 1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, // 3
	-1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, // 4
	-1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // 5
	 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, // 6
	 1.0f, -1.0f,  1.0f, 1.0f, 0.0f, 1.0f // 7
};

float vertices2[64] = {
	//positions
	-1.0f, -1.0f, -1.0f, // 0
	-1.0f,  1.0f, -1.0f, // 1
	 1.0f,  1.0f, -1.0f, // 2
	 1.0f, -1.0f, -1.0f, // 3
	-1.0f, -1.0f,  1.0f, // 4
	-1.0f,  1.0f,  1.0f, // 5
	 1.0f,  1.0f,  1.0f, // 6
	 1.0f, -1.0f,  1.0f, // 7

	 //color
	 0.0f, 0.0f, 0.0f,
	 0.0f, 1.0f, 0.0f,
	 1.0f, 1.0f, 0.0f,
	 1.0f, 0.0f, 0.0f,
	 0.0f, 0.0f, 1.0f,
	 0.0f, 1.0f, 1.0f,
	 1.0f, 1.0f, 1.0f,
	 1.0f, 0.0f, 1.0f,

	 //uvs
	0.f,1.f,
	0.f,0.f,
	1.f,0.f,
	1.f,1.f,

	0.f,0.f,
	0.f,1.f,
	1.f,1.f,
	1.f,0.f

};

WORD indicies[36] =
{
	0, 1, 2, 0, 2, 3,
	4, 6, 5, 4, 7, 6,
	4, 5, 1, 4, 1, 0,
	3, 2, 6, 3, 6, 7,
	1, 5, 6, 1, 6, 2,
	4, 0, 3, 4, 3, 7
};

typedef struct RenderLoadedAssets {
	ID3D12PipelineState* pipelineState;
	ID3D12RootSignature* rootSignature;
	unsigned short numVertexViews;
	D3D12_VERTEX_BUFFER_VIEW* vertexBufferViews;
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	ID3D12DescriptorHeap* tDescHeap;
}RenderAssets;

DirectX::XMMATRIX mvpMatrix;
void OnRender(RenderLoadedAssets& loadedAssets, DirectXRenderStruct& renderer)
{
	PIXScopedEvent(PIX_COLOR_INDEX(1), "Rendering");

	//static bool first = true;
	//static ID3D12GraphicsCommandList2* bList = nullptr;
	//
	//mvpMatrix = DirectX::XMMatrixMultiply(modelMatrix, viewMatrix);
	//mvpMatrix = XMMatrixMultiply(mvpMatrix, projectionMatrix);
	//if (first)
	//{
	//	first = false;
	//	ID3D12CommandAllocator* bAllocator = CreateCommandAllocator(renderer.device, D3D12_COMMAND_LIST_TYPE_BUNDLE);
	//	renderer.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bAllocator, nullptr, IID_PPV_ARGS(&bList));
	//	bList->SetPipelineState(loadedAssets.pipelineState);
	//	//bList->SetGraphicsRootSignature(loadedAssets.rootSignature);
	//	////bind the matrix to the GPU
	//	//bList->SetGraphicsRoot32BitConstants(0, sizeof(DirectX::XMMATRIX) / 4, &mvpMatrix, 0);
	//	bList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	//	bList->IASetVertexBuffers(0, loadedAssets.numVertexViews, loadedAssets.vertexBufferViews);
	//	bList->IASetIndexBuffer(&loadedAssets.indexBufferView);
	//	//Draw Command
	//	bList->DrawIndexedInstanced(_countof(indicies), 1, 0, 0, 0);
	//	bList->Close();
	//}

	GraphicsCommandQueue& cQueue = renderer.cQueue;
	ID3D12GraphicsCommandList2* cList = nullptr;
	ID3D12CommandAllocator* cAllocator = nullptr;
	if (!cQueue.cAllocators.Empty() && cQueue.fence.iFence->GetCompletedValue() >= cQueue.cAllocators.Front().fenceValue)
	{
		cAllocator = cQueue.cAllocators.Front().cAllocator;
		cQueue.cAllocators.Pop();
		cAllocator->Reset();
	}
	else
		cAllocator = CreateCommandAllocator(renderer.device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	if (!cQueue.cLists.Empty())
	{
		cList = cQueue.cLists.Front();
		cQueue.cLists.Pop();
		cList->Reset(cAllocator, nullptr);
	}
	else
		renderer.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cAllocator, nullptr, IID_PPV_ARGS(&cList));

	unsigned int currentBackBufferIndex = renderer.sChain->GetCurrentBackBufferIndex();
	ID3D12Resource* backBuffer = renderer.backBuffers[currentBackBufferIndex];
	//TODO: poder guradar la data del increment size per no preguntarho cada cop
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(renderer.rtvDescHeap->GetCPUDescriptorHandleForHeapStart(),
		currentBackBufferIndex, renderer.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV));
	D3D12_CPU_DESCRIPTOR_HANDLE dsv = renderer.dsvDescHeap->GetCPUDescriptorHandleForHeapStart();
	
	// Clear the render targets.
	{
		//Implicit State Transitions (https://docs.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
		//!!! cuidado si no executem aixo en + de una command list en la mateixa ExecuteCommandLists call
		// la debug layer em dona error si no poso la barrier???
		TransitionResource(cList, backBuffer,
			D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
	
		FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
	
		cList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
		cList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);

	}
	cList->SetPipelineState(loadedAssets.pipelineState);
	cList->SetGraphicsRootSignature(loadedAssets.rootSignature);
	ID3D12DescriptorHeap* ppHeaps[] = { loadedAssets.tDescHeap };
	cList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	cList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cList->IASetVertexBuffers(0, loadedAssets.numVertexViews, loadedAssets.vertexBufferViews);
	cList->IASetIndexBuffer(&loadedAssets.indexBufferView);
	//TODO: viewport glob variable that resizes on the window resize !!!
	cList->RSSetViewports(1, &renderer.viewport);
	cList->RSSetScissorRects(1, &renderer.scissorRect);
	//bound render targets to the Output merger
	cList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

	// Update the MVP matrix
	DirectX::XMMATRIX mvpMatrix = DirectX::XMMatrixMultiply(modelMatrix, viewMatrix);
	mvpMatrix = XMMatrixMultiply(mvpMatrix, projectionMatrix);
	//bind the matrix to the GPU
	cList->SetGraphicsRoot32BitConstants(0, sizeof(DirectX::XMMATRIX) / 4, &mvpMatrix, 0);
	cList->SetGraphicsRootDescriptorTable(1, loadedAssets.tDescHeap->GetGPUDescriptorHandleForHeapStart());

	//Draw Command
	cList->DrawIndexedInstanced(_countof(indicies), 1, 0, 0, 0);

	////using the bundle
	//cList->ExecuteBundle(bList);
	// Present
	{
		TransitionResource(cList, backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		//execute the cList
		//fenceValues[currentBackBufferIndex] = directCommandQueue->ExecuteCommandList(commandList);
		cList->Close();
		ID3D12CommandList* const ppCommandLists[] = {cList};
		renderer.cQueue.iCQueue->ExecuteCommandLists(1, ppCommandLists);
		renderer.rtvFenceValues[currentBackBufferIndex] = ++cQueue.fence.value;
		cQueue.iCQueue->Signal(cQueue.fence.iFence, cQueue.fence.value);
		cQueue.cAllocators.Push({ cAllocator, cQueue.fence.value });
		cQueue.cLists.Push(cList);
		//present the frame
		unsigned int sync = renderer.vSync ? 1 : 0;
		unsigned int presentFlags = (renderer.tearingSupport && !renderer.vSync) ? DXGI_PRESENT_ALLOW_TEARING : 0;
		renderer.sChain->Present(sync, presentFlags);
		currentBackBufferIndex = renderer.sChain->GetCurrentBackBufferIndex();
		unsigned long long fValue = renderer.rtvFenceValues[currentBackBufferIndex];
		//wait for fence value
		if (cQueue.fence.iFence->GetCompletedValue() < fValue)
		{
			cQueue.fence.iFence->SetEventOnCompletion(fValue, cQueue.fence.event);
			WaitForSingleObject(cQueue.fence.event, INFINITE);
			PIXNotifyWakeFromFenceSignal(cQueue.fence.event);
		}
	}
}

#include "dxcapi.h"
#include "d3d12shader.h"
#define COMPILED_SHADERS_PATH "CompiledShaders/"
#define COMPILED_SHADERS_EXT ".cso"
#define COMPILED_ROOT_SIGNATURE_EXT ".sig"
#define DEBUG_SHADERS_PATH "ShaderDebug/"
DxcCreateInstanceProc DxcCreateInstancePtr = nullptr;
HMODULE ShaderCompilerMod = nullptr;
void LoadShaderCompiler()
{
	ShaderCompilerMod = LoadLibrary("dxcompiler.dll");
	if (ShaderCompilerMod)
		DxcCreateInstancePtr = (DxcCreateInstanceProc)GetProcAddress(ShaderCompilerMod, "DxcCreateInstance");

	CreateDirectory(COMPILED_SHADERS_PATH, NULL);
	CreateDirectory(DEBUG_SHADERS_PATH, NULL);
}

IDxcBlob* CompileShader(LPCWSTR* compilerArguments, DWORD argCount, void* source, SIZE_T sourceSize, IDxcBlob** serializedRootSignature = nullptr, char* fileName = nullptr, char* fileExt = nullptr)
{
	if (!DxcCreateInstancePtr || !source)
		return nullptr;

	HRESULT hr;

	IDxcUtils* utils;
	//TODO: CLSID_DxcUtils is the normal here but it doesen't work so I wrote CLSID_DxcLibrary that seems to be the same
	hr = DxcCreateInstancePtr(CLSID_DxcLibrary, IID_PPV_ARGS(&utils));

	IDxcIncludeHandler* includeHandler;
	hr = utils->CreateDefaultIncludeHandler(&includeHandler);

	IDxcCompiler3* compiler3;
	hr = DxcCreateInstancePtr(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler3));

	IDxcResult* result;
	DxcBuffer sBuffer = {};
	sBuffer.Ptr = source;
	sBuffer.Size = sourceSize;
	sBuffer.Encoding = 0;
	compiler3->Compile(&sBuffer, compilerArguments, argCount, includeHandler, IID_PPV_ARGS(&result));

	result->GetStatus(&hr);

	includeHandler->Release();
	compiler3->Release();
	//Assumes default utf8 encoding, use IDxcUtf16 with -encoding utf16
	IDxcBlobUtf8* errorMsgs = nullptr;
	result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errorMsgs), nullptr);
	if (errorMsgs && errorMsgs->GetStringLength())
	{
		//TODO: LOG("Compiled returned HRESULT (0x%x), errors/warnings:\n\n %s\n",hr,errorMsgs->GetStringPointer())
		OutputDebugString(errorMsgs->GetStringPointer());
	}
	errorMsgs->Release();
	if (FAILED(hr)) {
		result->Release();
		return nullptr;
	}

	IDxcBlob* shaderObj;
	hr = result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderObj), nullptr);
	//TODO: save the compiled shader to a file and retreived later to consume on the engine
	if (fileName && fileExt) {
		char pathBuffer[256];
		unsigned int nameSize = StrLen(fileName);
		unsigned int extSize = StrLen(fileExt);

		//TODO: group construct path names on functions
		memcpy(pathBuffer, COMPILED_SHADERS_PATH, sizeof(COMPILED_SHADERS_PATH));
		memcpy(pathBuffer + (sizeof(COMPILED_SHADERS_PATH) - 1), fileName, nameSize - extSize);
		memcpy(pathBuffer + (sizeof(COMPILED_SHADERS_PATH) - 1) + (nameSize - extSize)-1, COMPILED_SHADERS_EXT, sizeof(COMPILED_SHADERS_EXT)-1);
		pathBuffer[sizeof(COMPILED_SHADERS_PATH) + (nameSize - extSize) + (sizeof(COMPILED_SHADERS_EXT) - 1) - 2] = '\0';
		hr = WriteBufferToFile(pathBuffer, shaderObj->GetBufferPointer(), shaderObj->GetBufferSize());

		if (serializedRootSignature)
		{
			hr = result->GetOutput(DXC_OUT_ROOT_SIGNATURE, IID_PPV_ARGS(serializedRootSignature), nullptr);
			memcpy(pathBuffer + (sizeof(COMPILED_SHADERS_PATH) - 1) + (nameSize - extSize) - 1, COMPILED_ROOT_SIGNATURE_EXT, sizeof(COMPILED_ROOT_SIGNATURE_EXT) - 1);
			pathBuffer[sizeof(COMPILED_SHADERS_PATH) + (nameSize - extSize) + (sizeof(COMPILED_ROOT_SIGNATURE_EXT) - 1) - 2] = '\0';
			hr = WriteBufferToFile(pathBuffer, (*serializedRootSignature)->GetBufferPointer(), (*serializedRootSignature)->GetBufferSize());
		}

		IDxcBlob* pdbData;
		//TODO: the pdb path is in utf16??
		IDxcBlobUtf16* pdbPathFromCompiler16;
		IDxcBlobUtf8* pdbPathFromCompiler;

		hr = result->GetOutput(DXC_OUT_PDB, IID_PPV_ARGS(&pdbData), &pdbPathFromCompiler16);
		utils->GetBlobAsUtf8(pdbPathFromCompiler16, &pdbPathFromCompiler);
		memcpy(pathBuffer, pdbPathFromCompiler->GetStringPointer() + 1, pdbPathFromCompiler->GetStringLength()-1);
		memcpy(pathBuffer + pdbPathFromCompiler->GetStringLength() - 2, fileName, nameSize);
		memcpy(pathBuffer + pdbPathFromCompiler->GetStringLength() - 2 + nameSize - extSize, "pdb", 4);
		//TODO: write the contents of the pdbData to a file at pdbPathFromCompiler (is important to save it with that name so that PIX can find it later)
		//hr = MyWriteBlobToFile(pdbData,pdbPathFromCompiler->GetStringPointer(),TRUE);
		hr = WriteBufferToFile(pathBuffer, pdbData->GetBufferPointer(), pdbData->GetBufferSize());
		pdbData->Release();
		pdbPathFromCompiler16->Release();
		pdbPathFromCompiler->Release();

		//reflaction data = data from the shader
		IDxcBlob* reflection;
		IDxcBlobUtf16* reflectionPathFromCompiler16;
		IDxcBlobUtf8* reflectionPathFromCompiler;
		hr = result->GetOutput(DXC_OUT_REFLECTION, IID_PPV_ARGS(&reflection), &reflectionPathFromCompiler16);
		utils->GetBlobAsUtf8(reflectionPathFromCompiler16, &reflectionPathFromCompiler);
		memcpy(pathBuffer,  reflectionPathFromCompiler->GetStringPointer() + 1, reflectionPathFromCompiler->GetStringLength());
		memcpy(pathBuffer + reflectionPathFromCompiler->GetStringLength() - 2, fileName, nameSize);
		memcpy(pathBuffer + reflectionPathFromCompiler->GetStringLength() - 2 + nameSize - extSize, "ref", 4);
		hr = WriteBufferToFile(pathBuffer, reflection->GetBufferPointer(), reflection->GetBufferSize());

		//DxcBuffer reflectionData = { reflection->GetBufferPointer(),reflection->GetBufferSize(),0U };
		//ID3D12ShaderReflection* D3D12Reflection;
		//hr = utils->CreateReflection(&reflectionData, IID_PPV_ARGS(&D3D12Reflection));
		////TODO: you can also save it to a file
		////TODO: how to use the reflection???
		//D3D12Reflection->Release();

		reflectionPathFromCompiler16->Release();
		reflectionPathFromCompiler->Release();
		reflection->Release();


	}

	result->Release();
	utils->Release();

	return shaderObj;
}

IDxcBlob* CompileShaderFromFile(const char* filePath, LPCWSTR* compilerArguments, DWORD argCount, IDxcBlob** serializedRootSignature = nullptr)
{
	IDxcBlob* shaderObj = nullptr;
	void* buffer = nullptr;
	DWORD bufferSize;
	HRESULT hr = ReadFileToBuffer(filePath, &buffer, &bufferSize);
	char* fileName = PointFilenameFromPath((char*)filePath);
	char* fileExt = PointExtensionFromPath((char*)filePath);
	shaderObj = CompileShader(compilerArguments, 9, buffer, bufferSize, serializedRootSignature, fileName, fileExt);
	HeapFree(GetProcessHeap(), HEAP_NO_SERIALIZE, buffer);
	return shaderObj;
}

struct ComputePipelineStateStream
{
	CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
	CD3DX12_PIPELINE_STATE_STREAM_CS             CS;
};
struct PipelineStateStream
{
	CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
	CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
	CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
	CD3DX12_PIPELINE_STATE_STREAM_VS VS;
	CD3DX12_PIPELINE_STATE_STREAM_PS PS;
	CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
	CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
} pipelineStateStream;

struct /*alignas(16)*/ GenerateMipsCB
{
	unsigned int      srcMipLevel;   // Texture level of source mip
	unsigned int      numMipLevels;  // Number of OutMips to write: [1-4]
	unsigned int      srcDimension;  // Width and height of the source texture are even or odd.
	unsigned int      isSRGB;        // Must apply gamma correction to sRGB textures.
	DirectX::XMFLOAT2 texelSize;     // 1.0 / OutMip1.Dimensions
};

RenderLoadedAssets LoadAssets(ID3D12Device2* device, ID3D12CommandQueue* cQueue, IWICImagingFactory* iFactory, bool* contentLoaded)
{
	void* textureFileBuffer = nullptr;
	DWORD textureFileBufferSize;
	const char* textureFilePath = "../assets/Directx9.png";
	HRESULT hr = ReadFileToBuffer(textureFilePath, &textureFileBuffer, &textureFileBufferSize);
	assert(SUCCEEDED(hr));
	TexMetadata tMetadata;
	Image* images = nullptr;
	unsigned char* imageMemory = nullptr;
	unsigned long long imageMemorySize = 0;
	unsigned long long nImages = 0;
	hr = WicLoadImageFromMemory(iFactory, textureFileBuffer, textureFileBufferSize, &tMetadata, images, nImages, imageMemory, imageMemorySize, 0);
	assert(SUCCEEDED(hr));
	D3D12_RESOURCE_DESC tDesc = {};
	switch (tMetadata.dimension)
	{
	case TEX_DIMENSION_TEXTURE1D:
		tDesc = CD3DX12_RESOURCE_DESC::Tex1D(tMetadata.format, (unsigned long long)tMetadata.width, (unsigned short)tMetadata.arraySize);
		break;
	case TEX_DIMENSION_TEXTURE2D:
		tDesc = CD3DX12_RESOURCE_DESC::Tex2D(tMetadata.format, (unsigned long long)tMetadata.width, (unsigned int)tMetadata.height, (unsigned short)tMetadata.arraySize);
		break;
	case TEX_DIMENSION_TEXTURE3D:
		tDesc = CD3DX12_RESOURCE_DESC::Tex3D(tMetadata.format, (unsigned long long)tMetadata.width, (unsigned int)tMetadata.height, (unsigned short)tMetadata.depth);
		break;
	default:
		assert(false);
		break;
	}

	ID3D12Resource* tResource = nullptr;
	CD3DX12_HEAP_PROPERTIES heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	//si a la D3D12_RESOURCE_DESC li passo mipmaps = 0 ell automaticament em posa els mips al resource i si li poso algun numero ell fa cas i crea aquell numero de mipmaps
	//tDesc.MipLevels = 1; //-> si li poso 1 ara funcionara sense haber de crear els mipmaps
	hr = device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &tDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tResource));
	assert(SUCCEEDED(hr));
	//to update the description to the actually created resource (the mips change from 0 to correct)
	tDesc = tResource->GetDesc();
	D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport; formatSupport.Format = tResource->GetDesc().Format;
	device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT));
	//https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12object-setname
	//Associates a name with the device object. This name is for use in debug diagnostics and tools
	tResource->SetName(L"First Texture");

	ID3D12DescriptorHeap* tDescHeap = nullptr;
	D3D12_DESCRIPTOR_HEAP_DESC tHeapDesc = {};
	tHeapDesc.NumDescriptors = 2 + tDesc.MipLevels-1 + 4;
	tHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	tHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	device->CreateDescriptorHeap(&tHeapDesc, IID_PPV_ARGS(&tDescHeap));
	CD3DX12_CPU_DESCRIPTOR_HANDLE tDescHeapHandle(tDescHeap->GetCPUDescriptorHandleForHeapStart());
	int incSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	
	//D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	//srvDesc.Format = tMetadata.format;
	////TODO: i si la txtura no es 2d...
	//srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	//srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	//srvDesc.Texture2D.MostDetailedMip = 0;
	//srvDesc.Texture2D.MipLevels = tMetadata.mipLevels;
	//srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
	//device->CreateShaderResourceView(tResource, &srvDesc, tDescHeap->GetCPUDescriptorHandleForHeapStart());
	//NULLPTR srvDesc => inherit texture resource format, dimensions, mips and array slices
	device->CreateShaderResourceView(tResource, nullptr, tDescHeapHandle);
	//setting the default UAV at the end of the mips (in case some outMips RETexture2D won't be used)
	tDescHeapHandle.Offset(incSize * (tDesc.MipLevels-1 + 2));
	for (unsigned int i = 0; i < 4; ++i)
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC defUavDesc = {};
		defUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		defUavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		defUavDesc.Texture2D.MipSlice = i;
		defUavDesc.Texture2D.PlaneSlice = 0;
		device->CreateUnorderedAccessView(nullptr, nullptr, &defUavDesc, tDescHeapHandle);
		tDescHeapHandle.Offset(incSize);
	}
	tDescHeapHandle.InitOffsetted(tDescHeap->GetCPUDescriptorHandleForHeapStart(), 1, incSize);

	// Create the PSO for GenerateMips shader.
	const char* mipsSrcPath = "../src/shaders/GenerateMips_CS.hlsl";
	void* mipsShaderBuffer = nullptr;
	DWORD mipsShaderBufferSize = 0;
	IDxcBlob* mipsShader = nullptr;
	IDxcBlob* mipsRootSignatureBlob = nullptr;
	void* mipsRootSignatureBuffer = nullptr;
	DWORD mipsRootSignatureBufferSize = 0;
	ID3D12RootSignature* mipsRootSignature;
	if (FileExists("CompiledShaders/GenerateMips_CS.cso") && (FileLastWriteTime("CompiledShaders/GenerateMips_CS.cso") > FileLastWriteTime(mipsSrcPath))) {
		ReadFileToBuffer("CompiledShaders/GenerateMips_CS.cso", &mipsShaderBuffer, &mipsShaderBufferSize);
		ReadFileToBuffer("CompiledShaders/GenerateMips_CS.sig", &mipsRootSignatureBuffer, &mipsRootSignatureBufferSize);
	}
	else {
		LPCWSTR pArguments[9] = { L"-Emain", L"-Tcs_6_6", L"-Qstrip_debug", L"-Zi", L"-Fd", L"\"ShaderDebug/\"", L"-Qstrip_reflect", L"-Fre", L"\"ShaderDebug/\"" };
		mipsShader = CompileShaderFromFile(mipsSrcPath, pArguments, 9, &mipsRootSignatureBlob);
		mipsShaderBuffer = mipsShader->GetBufferPointer();
		mipsShaderBufferSize = mipsShader->GetBufferSize();
		mipsRootSignatureBuffer = mipsRootSignatureBlob->GetBufferPointer();
		mipsRootSignatureBufferSize = mipsRootSignatureBlob->GetBufferSize();
	}
	device->CreateRootSignature(0, mipsRootSignatureBuffer, mipsRootSignatureBufferSize, IID_PPV_ARGS(&mipsRootSignature));

	ComputePipelineStateStream mipsPipelineStateStream;
	mipsPipelineStateStream.pRootSignature = mipsRootSignature;
	mipsPipelineStateStream.CS = CD3DX12_SHADER_BYTECODE(mipsShaderBuffer, mipsShaderBufferSize);
	D3D12_PIPELINE_STATE_STREAM_DESC mipsPipelineStateStreamDesc = { sizeof(ComputePipelineStateStream), &mipsPipelineStateStream };
	ID3D12PipelineState* mipsPipelineState = nullptr;
	device->CreatePipelineState(&mipsPipelineStateStreamDesc, IID_PPV_ARGS(&mipsPipelineState));
	//TODO: :(
	if (mipsShader)
	{
		mipsShader->Release();
		mipsRootSignatureBlob->Release();
	}
	else
	{
		HeapFree(GetProcessHeap(), 0, mipsShaderBuffer);
		HeapFree(GetProcessHeap(), 0, mipsRootSignatureBuffer);
	}

	//TODO: posar la command list i el seu allocator a la queue de command lists !!!!
	ID3D12CommandAllocator* cAllocator = CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	ID3D12GraphicsCommandList1* cList = CreateGraphicsCommandList(device, cAllocator, mipsPipelineState, D3D12_COMMAND_LIST_TYPE_DIRECT);
	//we can't release the pipelineState while the command list is executing or recording TODO:(queda pendent de esborrar)
	//mipsPipelineState->Release();

	D3D12_SUBRESOURCE_DATA* subresourceDataChunk = (D3D12_SUBRESOURCE_DATA*)HeapAlloc(GetProcessHeap(), 0, sizeof(D3D12_SUBRESOURCE_DATA) * nImages);
	for (int i = 0; i < nImages; ++i)
	{
		D3D12_SUBRESOURCE_DATA& subresourceData = subresourceDataChunk[i];
		subresourceData.pData = images[i].pixels;
		subresourceData.RowPitch = images[i].rowPitch;
		subresourceData.SlicePitch = images[i].slicePitch;
	}
	
	//We also use the command list to create and upload the texture
	ID3D12Resource* intermediateTextureUpload = nullptr;
	heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	unsigned long long requiredSize = 0;
	device->GetCopyableFootprints(&tDesc, 0, nImages, 0, nullptr, nullptr, nullptr, &requiredSize);
	CD3DX12_RESOURCE_DESC tUploadDesc = CD3DX12_RESOURCE_DESC::Buffer(requiredSize);
	device->CreateCommittedResource(&heapProperties,D3D12_HEAP_FLAG_NONE,&tUploadDesc,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&intermediateTextureUpload));
	UpdateSubresources(cList,tResource, intermediateTextureUpload, 0, 0, nImages, subresourceDataChunk);

	//We mainly use the command list to generate the mips (that is what the Pipeline State is set for)
	if (nImages < tDesc.MipLevels)
	{
		//multi sample or non 2d textures not allowed 
		if (!(tDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || tDesc.DepthOrArraySize != 1 || tDesc.SampleDesc.Count > 1))
		{
			ID3D12Resource* aliasResource = nullptr;
			ID3D12Resource* uavResource = tResource;
			//aquest format support sturcture es per cada textura
			D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport; formatSupport.Format = tDesc.Format;
			device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT));
	
			if (!(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
				(formatSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) &&
				(formatSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
				|| ((tDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0))
			{
				D3D12_RESOURCE_DESC aliasDesc = tDesc;
				aliasDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
				aliasDesc.Flags &= ~(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
	
				D3D12_RESOURCE_DESC uavDesc = aliasDesc;
				uavDesc.Format = GetUAVCompatableFormat(tDesc.Format);
	
				D3D12_RESOURCE_DESC resoruceDescs[] = { aliasDesc, uavDesc };
				D3D12_RESOURCE_ALLOCATION_INFO allocationInfo = device->GetResourceAllocationInfo(0, _countof(resoruceDescs), resoruceDescs);
				D3D12_HEAP_DESC heapDesc = {};
				heapDesc.SizeInBytes = allocationInfo.SizeInBytes;
				heapDesc.Alignment = allocationInfo.Alignment;
				heapDesc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
				heapDesc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
				heapDesc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
				heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
				//TODO: Track and destroy the heap and aliased resources when unused
				ID3D12Heap* heap;
				device->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));
	
				device->CreatePlacedResource(heap, 0, &aliasDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&aliasResource));
				device->CreatePlacedResource(heap, 0, &uavDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&uavResource));
	
				D3D12_RESOURCE_BARRIER barriers[] = {
					CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, aliasResource),
					//aquest barrier no cal pk la primera transition desde el common state de 1 resource es fa sola
					//CD3DX12_RESOURCE_BARRIER::Transition(aliasResource,D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES),
					//recordar que la tResource comença amb state copy_dest pk ha cambiat sol del common al copy_dest quan hem fet el upload heap
					CD3DX12_RESOURCE_BARRIER::Transition(tResource,D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
				};
				cList->ResourceBarrier(_countof(barriers), barriers);
				cList->CopyResource(aliasResource, tResource);
				//TODO: handle this alias back grouped with the next barriers grouped in just 1 list::resourceBarrier() call
				CD3DX12_RESOURCE_BARRIER tempAliasBarrier = CD3DX12_RESOURCE_BARRIER::Aliasing(aliasResource, uavResource);
				cList->ResourceBarrier(1, &tempAliasBarrier);
			}
	
			// Create a SRV that uses the format of the original texture.
			CD3DX12_RESOURCE_DESC uavResourceDesc(uavResource->GetDesc());
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Format = IsSRGBFormat(tDesc.Format) ? GetSRGBFormat(uavResourceDesc.Format) : uavResourceDesc.Format;
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srvDesc.ViewDimension =
				D3D12_SRV_DIMENSION_TEXTURE2D;  // Only 2D textures are supported (this was checked in the calling function).
			srvDesc.Texture2D.MipLevels = uavResourceDesc.MipLevels;
			device->CreateShaderResourceView(uavResource, nullptr, tDescHeapHandle);
			tDescHeapHandle.Offset(incSize);

			//create the UAViews (el descriptor heap ia li he posat prou espai mes adalt)
			for (unsigned int i = 1; i < tDesc.MipLevels; ++i)
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.Format = uavResourceDesc.Format;
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				uavDesc.Texture2D.PlaneSlice = 0;
				uavDesc.Texture2D.MipSlice = i;
				device->CreateUnorderedAccessView(uavResource, nullptr, &uavDesc, tDescHeapHandle);
				tDescHeapHandle.Offset(incSize);
			}

			cList->SetComputeRootSignature(mipsRootSignature);
			//TODO: es pot borrar la root signature despres de setejarla si ja no la necesitem mes???
			mipsRootSignature->Release();
			cList->SetDescriptorHeaps(1, &tDescHeap);

			CD3DX12_GPU_DESCRIPTOR_HANDLE srvGPUHandle(tDescHeap->GetGPUDescriptorHandleForHeapStart(), 1, incSize);
			CD3DX12_GPU_DESCRIPTOR_HANDLE tGPUDescHeapHandle(tDescHeap->GetGPUDescriptorHandleForHeapStart(), 2, incSize);
			GenerateMipsCB mipsCB;
			mipsCB.isSRGB = IsSRGBFormat(uavResourceDesc.Format);
			for (unsigned int srcMip = 0; srcMip < uavResourceDesc.MipLevels - 1u; )
			{
				unsigned long long srcWidth = uavResourceDesc.Width >> srcMip;
				unsigned int srcHeight = uavResourceDesc.Height >> srcMip;
				unsigned int dstWidth = (unsigned int)(srcWidth >> 1);
				unsigned int dstHeight = srcHeight >> 1;

				// 0b00(0): Both width and height are even.
				// 0b01(1): Width is odd, height is even.
				// 0b10(2): Width is even, height is odd.
				// 0b11(3): Both width and height are odd.
				mipsCB.srcDimension = (srcHeight & 1) << 1 | (srcWidth & 1);

				// How many mipmap levels to compute this pass (max 4 mips per pass)
				DWORD mipCount;
				// The number of times we can half the size of the texture and get
				// exactly a 50% reduction in size.
				// A 1 bit in the width or height indicates an odd dimension.
				// The case where either the width or the height is exactly 1 is handled
				// as a special case (as the dimension does not require reduction).
				_BitScanForward(&mipCount, (dstWidth == 1 ? dstHeight : dstWidth) | (dstHeight == 1 ? dstWidth : dstHeight));
				// Maximum number of mips to generate is 4.
				mipCount = (mipCount + 1) < 4 ? (mipCount + 1) : 4; //std::min<DWORD>(4, mipCount + 1); 
				// Clamp to total number of mips left over.
				mipCount = (srcMip + mipCount) >= uavResourceDesc.MipLevels ? uavResourceDesc.MipLevels - srcMip - 1 : mipCount;

				// Dimensions should not reduce to 0.
				// This can happen if the width and height are not the same.
				dstWidth = 1 < dstWidth ? dstWidth : 1; //std::max<DWORD>(1, dstWidth); 
				dstHeight = 1 < dstHeight ? dstHeight : 1; //std::max<DWORD>(1, dstHeight);

				mipsCB.srcMipLevel = srcMip;
				mipsCB.numMipLevels = mipCount;
				mipsCB.texelSize.x = 1.0f / (float)dstWidth;
				mipsCB.texelSize.y = 1.0f / (float)dstHeight;

				cList->SetComputeRoot32BitConstants(0, sizeof(mipsCB) / 4, &mipsCB, 0);
				//without this barrier resource can't be binded in a non pixel shader!!
				//D3D12_RESOURCE_BARRIER barriers[] = {
				//	CD3DX12_RESOURCE_BARRIER::Transition(uavResource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
				//};
				//cList->ResourceBarrier(_countof(barriers), barriers);
				cList->SetComputeRootDescriptorTable(1, srvGPUHandle);
				//DIRIA K NO CAL PK ELS UAV els em creat mes adalt
				//for (unsigned int mip = 0; mip < mipCount; ++mip)
				//{
				//	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				//	uavDesc.Format = uavResourceDesc.Format;
				//	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				//	uavDesc.Texture2D.MipSlice = srcMip + mip + 1;
				//
				//	device->CreateUnorderedAccessView(uavResource, nullptr, &uavDesc, )
				//	auto uav = m_Device.CreateUnorderedAccessView(texture, nullptr, &uavDesc);
				//	SetUnorderedAccessView(GenerateMips::OutMip, mip, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				//		srcMip + mip + 1, 1);
				//}
				cList->SetComputeRootDescriptorTable(2, tGPUDescHeapHandle);
				tGPUDescHeapHandle.Offset(incSize * mipCount);

				cList->Dispatch((dstWidth + 8 - 1) / 8, (dstHeight + 8 - 1) / 8, 1);
				//TODO: aquestes barriers es poden ajuntar en 1 sola call a ResourceBarrier() fora del for
				D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(uavResource);
				cList->ResourceBarrier(1, &barrier);
				srcMip += mipCount;
			}

			if (aliasResource)
			{
				D3D12_RESOURCE_BARRIER barriers[] = { CD3DX12_RESOURCE_BARRIER::Aliasing(uavResource, aliasResource),
					CD3DX12_RESOURCE_BARRIER::Transition(tResource,D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES),
					CD3DX12_RESOURCE_BARRIER::Transition(aliasResource,D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
				};
				cList->ResourceBarrier(_countof(barriers), barriers);
				cList->CopyResource(tResource, aliasResource);
				//TODO: cal tornar a posar el resource tResource a state common o despres the executarse la list ia torna al state original???
			}
		}
	}
	HeapFree(GetProcessHeap(), 0, subresourceDataChunk);
	D3D12_RESOURCE_DESC desccc = tResource->GetDesc();

	//We also use the cList to transfer the vertex and index buffers
	ID3D12Resource* vertexBuffer = nullptr;
	ID3D12Resource* intermediateVertexBuffer = nullptr;
	//UpdateBufferResource(device, cList, &vertexBuffer, &intermediateVertexBuffer, vertices, _countof(vertices), sizeof(float), D3D12_RESOURCE_FLAG_NONE);
	UpdateBufferResource(device, cList, &vertexBuffer, &intermediateVertexBuffer, vertices2, _countof(vertices2), sizeof(float), D3D12_RESOURCE_FLAG_NONE);
	ID3D12Resource* indexBuffer;
	ID3D12Resource* intermediateIndexBuffer;
	UpdateBufferResource(device, cList, &indexBuffer, &intermediateIndexBuffer, indicies, _countof(indicies), sizeof(WORD), D3D12_RESOURCE_FLAG_NONE);
	//TODO: If i don't transition it breaks on the renderer
	CD3DX12_RESOURCE_BARRIER tempTransitionBarrier = CD3DX12_RESOURCE_BARRIER::Transition(tResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
	cList->ResourceBarrier(1, &tempTransitionBarrier);
	cList->Close();
	ID3D12CommandList* ppCommandLists[] = {cList};
	cQueue->ExecuteCommandLists(1 , ppCommandLists);
	cList->Release();
	//TODO: release the allocator when finished the execution of the command list
	//cAllocator->Release();

	//to tell the input assembler where the vertices are on GPU memory and how their buffer(s) is
	//we will call ID3D12GraphicsCommandList::IASetVertexBuffers() using this structure
	//D3D12_VERTEX_BUFFER_VIEW vertexBufferView = {};
	//vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
	//vertexBufferView.SizeInBytes = sizeof(vertices);
	//vertexBufferView.StrideInBytes = 6 * sizeof(float);

	unsigned short numVertexBufferViews = 3;
	D3D12_VERTEX_BUFFER_VIEW* vertexBufferViews = (D3D12_VERTEX_BUFFER_VIEW*)HeapAlloc(GetProcessHeap(), 0, sizeof(D3D12_VERTEX_BUFFER_VIEW) * numVertexBufferViews);
	vertexBufferViews[0].BufferLocation = vertexBuffer->GetGPUVirtualAddress();
	vertexBufferViews[0].SizeInBytes = 24 * sizeof(float);
	vertexBufferViews[0].StrideInBytes = sizeof(float) * 3;

	vertexBufferViews[1].BufferLocation = vertexBuffer->GetGPUVirtualAddress() + sizeof(float) * 24;
	vertexBufferViews[1].SizeInBytes = 24 * sizeof(float);
	vertexBufferViews[1].StrideInBytes = sizeof(float) * 3;

	vertexBufferViews[2].BufferLocation = vertexBuffer->GetGPUVirtualAddress() + sizeof(float) * 48;
	vertexBufferViews[2].SizeInBytes = 16 * sizeof(float);
	vertexBufferViews[2].StrideInBytes = sizeof(float) * 2;

	//to tell the input assembler where the indices are on GPU memory and how their buffer is
	//we will call ID3D12GraphicsCommandList::IASetIndexBuffer() using this structure
	D3D12_INDEX_BUFFER_VIEW indexBufferView = {};
	indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
	indexBufferView.SizeInBytes = sizeof(indicies);
	indexBufferView.Format = DXGI_FORMAT_R16_UINT;

	//Creating the PSO for the rendering later
	ID3D12RootSignature* rootSignature = CreateRootSignature(device);

	const char* vertexPath = "../src/shaders/VertexShader.hlsl";
	IDxcBlob* vertexShader = nullptr;
	void* vertexShaderBuffer = nullptr;
	DWORD vertexShaderBufferSize = 0;
	if (FileExists("CompiledShaders/VertexShader.cso") && (FileLastWriteTime("CompiledShaders/VertexShader.cso") > FileLastWriteTime(vertexPath))) {
		ReadFileToBuffer("CompiledShaders/VertexShader.cso", &vertexShaderBuffer, &vertexShaderBufferSize);
	}
	else {
		LPCWSTR vArguments[9] = { L"-Emain", L"-Tvs_6_6", L"-Qstrip_debug", L"-Zi", L"-Fd", L"\"ShaderDebug/\"", L"-Qstrip_reflect", L"-Fre", L"\"ShaderDebug/\"" };
		vertexShader = CompileShaderFromFile(vertexPath, vArguments, 9);
		vertexShaderBuffer = vertexShader->GetBufferPointer();
		vertexShaderBufferSize = vertexShader->GetBufferSize();
	}
	const char* pixelPath = "../src/shaders/PixelShader.hlsl";
	IDxcBlob* pixelShader = nullptr;
	void* pixelShaderBuffer = nullptr;
	DWORD pixelShaderBufferSize = 0;
	if (FileExists("CompiledShaders/PixelShader.cso") && (FileLastWriteTime("CompiledShaders/PixelShader.cso") > FileLastWriteTime(pixelPath))) {
		ReadFileToBuffer("CompiledShaders/PixelShader.cso", &pixelShaderBuffer, &pixelShaderBufferSize);
	}
	else {
		LPCWSTR pArguments[9] = { L"-Emain", L"-Tps_6_6", L"-Qstrip_debug", L"-Zi", L"-Fd", L"\"ShaderDebug/\"", L"-Qstrip_reflect", L"-Fre", L"\"ShaderDebug/\"" };
		pixelShader = CompileShaderFromFile(pixelPath, pArguments, 9);
		pixelShaderBuffer = pixelShader->GetBufferPointer();
		pixelShaderBufferSize = pixelShader->GetBufferSize();
	}


	// Create the vertex input layout
	D3D12_INPUT_ELEMENT_DESC inputLayoutDesc[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "UV", 0, DXGI_FORMAT_R32G32_FLOAT, 2, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	//Com o fan al tuto
	D3D12_RT_FORMAT_ARRAY rtvFormats = {};
	rtvFormats.NumRenderTargets = 1;
	rtvFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	pipelineStateStream.pRootSignature = rootSignature;
	pipelineStateStream.InputLayout = { inputLayoutDesc, _countof(inputLayoutDesc) };
	pipelineStateStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	pipelineStateStream.VS = CD3DX12_SHADER_BYTECODE(vertexShaderBuffer, vertexShaderBufferSize);
	pipelineStateStream.PS = CD3DX12_SHADER_BYTECODE(pixelShaderBuffer, pixelShaderBufferSize);
	pipelineStateStream.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	pipelineStateStream.RTVFormats = rtvFormats;
	D3D12_PIPELINE_STATE_STREAM_DESC pipelineStateStreamDesc = {
		sizeof(PipelineStateStream), &pipelineStateStream
	};
	//TODO: Canvio totes les interfaces de device a device2???
	ID3D12PipelineState* pipelineState = nullptr;
	device->CreatePipelineState(&pipelineStateStreamDesc, IID_PPV_ARGS(&pipelineState));

	//TODO: :(
	if (vertexShader)
		vertexShader->Release();
	else
		HeapFree(GetProcessHeap(), 0, vertexShaderBuffer);
	if (pixelShader)
		pixelShader->Release();
	else
		HeapFree(GetProcessHeap(), 0, pixelShaderBuffer);

	// Describe and create the graphics pipeline state object (PSO).
	//ID3D12PipelineState* pipelineState = nullptr;
	//D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
	//psoDesc.InputLayout = { inputLayoutDesc, _countof(inputLayoutDesc) };
	//psoDesc.pRootSignature = rootSignature;
	//psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
	//psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
	//psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	//psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	//psoDesc.DepthStencilState.DepthEnable = FALSE;
	//psoDesc.DepthStencilState.StencilEnable = FALSE;
	//psoDesc.SampleMask = UINT_MAX;
	//psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	//psoDesc.NumRenderTargets = 1;
	//psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	//psoDesc.SampleDesc.Count = 1;
	//device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState));

	*contentLoaded = true;

	return { pipelineState, rootSignature, numVertexBufferViews, vertexBufferViews, indexBufferView, tDescHeap };
}


//This init is before the window creation. After creating the window we still need to create the swap chain and the back buffers
inline void InitRenderer(DirectXRenderStruct& renderer) 
{
	HMODULE hD3D12Library = LoadLibraryA("D3d12.dll");

	//TODO: exit program or big error if not access (s'ha de fer error handeling per tot el programa)
	if (hD3D12Library)
	{
#ifdef _DEBUG
		D3D12GetDebugInterfacePtr = (PFN_D3D12_GET_DEBUG_INTERFACE)GetProcAddress(hD3D12Library, "D3D12GetDebugInterface");
#endif // _DEBUG
		D3D12CreateDevicePtr = (PFN_D3D12_CREATE_DEVICE)GetProcAddress(hD3D12Library, "D3D12CreateDevice");
		D3D12SerializeRootSignaturePtr = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)GetProcAddress(hD3D12Library, "D3D12SerializeRootSignature");
		D3D12SerializeVersionedRootSignaturePtr = (PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE)GetProcAddress(hD3D12Library, "D3D12SerializeVersionedRootSignature");
	}
	HMODULE hDXGILibrary = LoadLibraryA("Dxgi.dll");
	if (hDXGILibrary)
	{
		CreateDXGIFactory1Ptr = (CREATE_DXGI_FACTORY1)GetProcAddress(hDXGILibrary, "CreateDXGIFactory1");
		CreateDXGIFactory2Ptr = (CREATE_DXGI_FACTORY2)GetProcAddress(hDXGILibrary, "CreateDXGIFactory2");
	}

	InitDebugLayer();
	renderer.adapter = GetAdapter(false);
	renderer.device = CreateDevice(renderer.adapter);
	renderer.cQueue.iCQueue = CreateCommandQueue(renderer.device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	renderer.cQueue.fence = {}; renderer.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&renderer.cQueue.fence.iFence));
	//renderer.cQueue.fence.value = 0; //it gets initialized to 0 with the = {}
	renderer.cQueue.fence.event = CreateEvent(NULL, FALSE, FALSE, NULL);;
	renderer.rtvFenceValues[0] = 0; renderer.rtvFenceValues[1] = 0; renderer.rtvFenceValues[2] = 0;
	// Create the descriptor heap for the depth-stencil view.
	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	renderer.device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&renderer.dsvDescHeap));

	renderer.viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, (float)windowClientRect.right, (float)windowClientRect.bottom);
	renderer.scissorRect = CD3DX12_RECT(0, 0, LONG_MAX, LONG_MAX);
}