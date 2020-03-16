#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>

int 
main(int argc, char **argv) {
	if (argc < 2) {
		printf("my-recogonition: expected image filename as argument\n");
		printf("example usage: ./my-recognition my_image.jpg\n");
		return 0;
	}
	const char *imgFilename = argv[1];
	
	// these variables will be used to store the image data and dimensions
	// the image data will be stored in shared CPU/GPU memory, so there are
	// pointers for the CPU and GPU (both reference the same physical memory)
	float *imgCPU = NULL; // CPU pointer to floating-point RGBA image data
	float *imgCUDA = NULL; // GPU pointer to floating-point RGBA image data
	int imgWidth = 0;
	int imgHeight = 0;
	if (!loadImageRGBA(imgFilename,(float4**)&imgCPU,(float4**)&imgCUDA,&imgWidth,&imgHeight)) {
		printf("failed to load image '%s'\n",imgFilename);
		return 0;
	}
	imageNet *net = imageNet::Create(imageNet::GOOGLENET);
	if (!net) {
		printf("failed to load image recognition network\n");
		return 0;
	}
	float confidence = 0.0;
	// classify the image with TensorRT on the GPU (hence we use the CUDA pointer)
	const int classIndex = net->Classify(imgCUDA,imgWidth,imgHeight,&confidence);
	if (classIndex >= 0) {
		const char *classDescription = net->GetClassDesc(classIndex);
		printf("image is recognized as '%s' (class#%i) with %f%% confidence\n",
				classDescription,classIndex,confidence*100.0f);
	} else {
		printf("failed to classify image\n");
	}
	delete net;
	return 0;
}

