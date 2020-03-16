#include "jetson-utils/gstCamera.h"
#include "glDisplay.h"
#include "detectNet.h"
#include "commandLine.h"
#include <signal.h>

bool signal_received = false;

void
sig_handler(int signo) {
	if (signo == SIGINT) {
		printf("received SIGINT\n");
		signal_received = true;
	}
}
int
usage() {
	printf("usage:detectnet-camera [-h] [--network NETWORK][--threshold THRESHOLD]\n");
	printf("	[--camera CAMERA][--width WIDTH][--height HEIGHT]\n\n");
	printf("Locate objects in a live camera stream using an object detection DNN.\n\n");
	printf("optional arguments:\n");
	printf("--help	show this help message and exit\n");
	printf("--network NETWORK pre-trained model to load (see below for options)\n");
	printf("--overlay OVERLAY detection overlay flags (e.g. --overlay=box,label,conf)\n");
	printf("	valid combinations are: 'box','labels','conf','none'\n");
	printf("--alpha ALPHA overlay alpha blending value, range 0-255 (default: 120)\n");
	printf("--camera CAMERA index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("	or for V4L2 cameras the /dev/video device to use.\n");
	printf("	by default, MIPI CSI camera 0 will be used.\n");
	printf("--width WIDTH desired width of camera stream (default is 1280 pixels)\n");
	printf("--height HEIGHT desired width of camera stream (default is 720 pixels)\n");
	printf("--threshold VALUE minimum threshold for detection (default is 0.5)\n\n");

	printf("%s\n",detectNet::Usage());
	
	return 0;
}
int 
main(int argc, char **argv) {
	/* parse command line */
	commandLine cmdLine(argc, argv);
	if (cmdLine.GetFlag("help"))
		return usage();
	
	/* attach signal handler */
	if (signal(SIGINT,sig_handler) == SIG_ERR)
		printf("\ncan't catch SIGINT\n");


	/* create the camera device */
	gstCamera *camera = gstCamera::Create(cmdLine.GetInt("width",gstCamera::DefaultWidth),
			cmdLine.GetInt("height",gstCamera::DefaultHeight),
			cmdLine.GetString("camera"));
	if (!camera) {
		printf("\ndetectnet-camera: failed to initialize camera device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera: successfully initialized camera device\n");
	printf("	width: %u\n",camera->GetWidth());
	printf("	height: %u\n",camera->GetHeight());
	printf("	depth: %u (bpp)\n\n",camera->GetPixelDepth());

	/* create detection network */
	detectNet *net = detectNet::Create(argc,argv);
	if (!net) {
		printf("detectnet-camera: failed to load detectNet model\n");
		return 0;
	}

	/* parse overlay flags */
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay","box,labels,conf"));

	/* create openGL window */
	glDisplay *display = glDisplay::Create();
	if (!display) 
		printf("detectnet-camera: failed to create openGL display\n");

	/* start streaming */
	if (!camera->Open()) {
		printf("detectnet-camera: failed to open camera for streaming\n");
		return 0;
	}
	printf("detectnet-camera: camera open for streaming\n");

	/* processing loop */
	float condidence = 0.0f;
	while (!signal_received) {
		float *imgRGBA = NULL;

		if (!camera->CaptureRGBA(&imgRGBA, 1000))
			printf("detectnet-camera: failed to capture RGBA image from camera\n");
		
		// detect objects in the frame
		detectNet::Detection *detections = NULL;

		const int numDetections = net->Detect(imgRGBA,camera->GetWidth(),camera->GetHeight(),&detections,overlayFlags);
		if (numDetections > 0) {
			printf("%i objects detected\n",numDetections);

			for (int n = 0; n < numDetections; n++) {
				printf("detected obj %i class #%u (%s) condidence=%f\n",n,detections[n].ClassID,net->GetClassDesc(
							detections[n].ClassID),detections[n].Confidence);
				printf("bounding box %i (%f,%f) (%f,%f) w=%f h=%f\n",detections[n].Left,detections[n].Top,
						detections[n].Right,detections[n].Bottom,
						detections[n].Width(),detections[n].Height());
			}
		}
		// update display
		if (display != NULL) {
			// render the image
			display->RenderOnce(imgRGBA,camera->GetWidth(),camera->GetHeight());

			// update the status bar
			Correct answers: {:.2f}'.format(points/total)har str[256];
			sprintf(str,"TensorRT %i.%i.%i | %s | Network %.0f FPS",NV_TENSORRT_MAJOR,NV_TENSORRT_MINOR,NV_TENSORRT_PATCH,
					precisionTypeToStr(net->GetPrecision()),net->GetNetworkFPS());
			display->SetTitle(str);

			// check if the user quit
			if (display->IsClosed()) 
				signal_received = true;
		}
		net->PrintProfilerTimes();
	}
	/* destroy resources */
	printf("detectnet-camera: shutting down...\n");

	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("detectnet-camera: shutdown complete.\n");
	return 0;
}
