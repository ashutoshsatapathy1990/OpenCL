//IMAGE ZOOM IN AND ZOOM OUT USING INTERPOLATION 

//Header files
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <complex.h>
#ifdef __APPLE__
#include <OpenCL\cl.h>
#else
#include <CL\cl.h>
#endif
#include <opencv\cv.h>
#include <opencv\highgui.h>
using namespace std;
using namespace cv;

// OpenCL Kernel which is run for every work item created
const char *Magnify_kernel =                                                          
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable							\n" \
"#pragma OPENCL EXTENSION cl_khr_printf : enable				                \n" \
"__kernel						 	                                \n" \
"void Magnify_kernel (__global int* source,					                \n" \
"	__global int* dst,								        \n" \
"	int height,									        \n" \
"	int width,										\n" \
"	int reheight,							                        \n" \
"	int rewidth)								                \n" \
"{											        \n" \
"	//Get the index of the work items				                        \n" \
"	uint globalId = get_global_id(0);				                        \n" \
"	float ty = ((float)(width-1)) / (rewidth -1);				        	\n" \
"	float tx = ((float)(height-1)) / (reheight -1);			        		\n" \
"	float x = tx * (globalId / rewidth);			                		\n" \
"	float y = ty * (globalId % rewidth);				                        \n" \
"	int x1 = floor(x);			      		                                \n" \
"	int x2 = ceil(x);					                                \n" \
"	int y1 = floor(y);					                                \n" \
"	int y2 = ceil(y);					                                \n" \
"	float ty1 = ((y1 == y2)? 1 : fabs(y2 - y));				                \n" \
"	float ty2 = ((y1 == y2)? 0 : fabs(y1 - y));				                \n" \
"	float tx1 = ((x1 == x2)? 1 : fabs(x2 - x));				                \n" \
"	float tx2 = ((x1 == x2)? 0 : fabs(x1 - x));			 	                \n" \
"	float rowx1 = ty1 * source[x1 * width + y1] + ty2 * source[x1 * width + y2];		\n" \
"	float rowx2 = ty1 * source[x2 * width + y1] + ty2 * source[x2 * width + y2];		\n" \
"	float coly = tx1 * rowx1 + tx2 * rowx2;				   	                \n" \
"	dst[reheight * (globalId / rewidth) + (globalId % rewidth)] = coly;			\n" \
"}											        \n" \
"\n";

//Main Program
int main(void)
{
	//Read image data from image file
	Mat Image = imread("#INPUT IMAGE PATH", CV_LOAD_IMAGE_COLOR);
	if (!Image.data)
	{
		printf("COULDN'T OPEN OR READ LENA.TIFF FILE \n");
		return -1;
	}
	namedWindow("LENA", CV_WINDOW_AUTOSIZE);
	imshow("LENA", Image);

	//Read separate channels (R, G, B) from an Image
	Mat RGB_Image[3];
	split(Image, RGB_Image);

	//Display separate scale (R, G, B) of an RGB image
	namedWindow("RED CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("RED CHANNEL", RGB_Image[0]);
	namedWindow("GREEN CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("GREEN CHANNEL", RGB_Image[1]);
	namedWindow("BLUE CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("BLUE CHANNEL", RGB_Image[2]);

	//Enter size of resized image (cols x rows)
	int row, col;
	printf("Enter the no.of rows and cols of final image \n");
	scanf("%d %d", &row, &col);

	//Create empty images of above entered size (cols x rows)
	Mat Final_Image, Final_RGB_Image[3];
	Final_Image.create(row, col, CV_8U);
	Final_RGB_Image[0].create(row, col, CV_8U);
	Final_RGB_Image[1].create(row, col, CV_8U);
	Final_RGB_Image[2].create(row, col, CV_8U);

	//Create host buffers of image size and final image size
	int* RED_Channel = (int*)malloc(sizeof(int) * Image.rows * Image.cols);
	int* GREEN_Channel = (int*)malloc(sizeof(int) * Image.rows * Image.cols);
	int* BLUE_Channel = (int*)malloc(sizeof(int) * Image.rows * Image.cols);
	int* Final_RED_Channel = (int*)malloc(sizeof(int) * Final_Image.rows * Final_Image.cols);
	int* Final_GREEN_Channel = (int*)malloc(sizeof(int) * Final_Image.rows * Final_Image.cols);
	int* Final_BLUE_Channel = (int*)malloc(sizeof(int) * Final_Image.rows * Final_Image.cols);

	//Copy image data to host buffer
	for (int i = 0; i < Image.cols * Image.rows; i++)
	{
		RED_Channel[i] = RGB_Image[0].data[i];
		GREEN_Channel[i] = RGB_Image[1].data[i];
		BLUE_Channel[i] = RGB_Image[2].data[i];
	}

	//Get the platform information
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms;

	//Set up the platforms
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

	//Get the device lists and choose the device you want to run on
	cl_device_id* device_list = NULL;
	cl_uint num_devices;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL);

	//Create an OPenCL Context for each device in the platform
	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

	//Create a command a queue
	cl_command_queue command_queue_magnify = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

	//Create OpenCL device buffer
	cl_mem RED_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Image.rows * Image.cols, RED_Channel, &clStatus);
	cl_mem GREEN_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Image.rows * Image.cols, GREEN_Channel, &clStatus);
	cl_mem BLUE_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Image.rows * Image.cols, BLUE_Channel, &clStatus);
	cl_mem Final_RED_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Final_Image.rows * Final_Image.cols, Final_RED_Channel, &clStatus);
	cl_mem Final_GREEN_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Final_Image.rows * Final_Image.cols, Final_GREEN_Channel, &clStatus);
	cl_mem Final_BLUE_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * Final_Image.rows * Final_Image.cols, Final_BLUE_Channel, &clStatus);

	//Create a program for the kernel source
	cl_program program_magnify = clCreateProgramWithSource(context, 1, (const char**)&Magnify_kernel, NULL, &clStatus);

	//Build the program
	clStatus = clBuildProgram(program_magnify, 1, device_list, NULL, NULL, NULL);

	//Create the OpenCL kernel
	cl_kernel kernel_magnify_RED = clCreateKernel(program_magnify, "Magnify_kernel", &clStatus);
	cl_kernel kernel_magnify_GREEN = clCreateKernel(program_magnify, "Magnify_kernel", &clStatus);
	cl_kernel kernel_magnify_BLUE = clCreateKernel(program_magnify, "Magnify_kernel", &clStatus);

	//Set the arguments of the kernels
	clStatus = clSetKernelArg(kernel_magnify_RED, 0, sizeof(cl_mem), (void *)&RED_clmem);
	clStatus = clSetKernelArg(kernel_magnify_RED, 1, sizeof(cl_mem), (void *)&Final_RED_clmem);
	clStatus = clSetKernelArg(kernel_magnify_RED, 2, sizeof(int), (void *)&Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_RED, 3, sizeof(int), (void *)&Image.cols);
	clStatus = clSetKernelArg(kernel_magnify_RED, 4, sizeof(int), (void *)&Final_Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_RED, 5, sizeof(int), (void *)&Final_Image.cols);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 0, sizeof(cl_mem), (void *)&GREEN_clmem);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 1, sizeof(cl_mem), (void *)&Final_GREEN_clmem);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 2, sizeof(int), (void *)&Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 3, sizeof(int), (void *)&Image.cols);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 4, sizeof(int), (void *)&Final_Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_GREEN, 5, sizeof(int), (void *)&Final_Image.cols);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 0, sizeof(cl_mem), (void *)&BLUE_clmem);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 1, sizeof(cl_mem), (void *)&Final_BLUE_clmem);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 2, sizeof(int), (void *)&Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 3, sizeof(int), (void *)&Image.cols);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 4, sizeof(int), (void *)&Final_Image.rows);
	clStatus = clSetKernelArg(kernel_magnify_BLUE, 5, sizeof(int), (void *)&Final_Image.cols);

	//Execute the kernels for zoom in or zoom out
	size_t global_size = Final_Image.rows * Final_Image.cols;
	size_t local_size = 1;
	clStatus = clEnqueueNDRangeKernel(command_queue_magnify, kernel_magnify_RED, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue_magnify, kernel_magnify_GREEN, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue_magnify, kernel_magnify_BLUE, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

	//Read device buffer to host buffer
	clStatus = clEnqueueReadBuffer(command_queue_magnify, Final_RED_clmem, CL_TRUE, 0, Final_Image.rows * Final_Image.cols * sizeof(int), Final_RED_Channel, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_magnify, Final_GREEN_clmem, CL_TRUE, 0, Final_Image.rows * Final_Image.cols * sizeof(int), Final_GREEN_Channel, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_magnify, Final_BLUE_clmem, CL_TRUE, 0, Final_Image.rows * Final_Image.cols * sizeof(int), Final_BLUE_Channel, 0, NULL, NULL);

	//Clean up and wait for all the commands to complete.
	clStatus = clFlush(command_queue_magnify);
	clStatus = clFinish(command_queue_magnify);

	//Copy host buffer to image buffer
	for (int i = 0; i < Final_Image.rows * Final_Image.cols; i++)
	{
		Final_RGB_Image[0].at<uchar>(i / Final_Image.cols, i % Final_Image.cols) = Final_RED_Channel[i];
		Final_RGB_Image[1].at<uchar>(i / Final_Image.cols, i % Final_Image.cols) = Final_GREEN_Channel[i];
		Final_RGB_Image[2].at<uchar>(i / Final_Image.cols, i % Final_Image.cols) = Final_BLUE_Channel[i];
	}

	//Display separate scale (R, G, B) of smoothening images
	namedWindow("RESIZED RED CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("RESIZED RED CHANNEL", Final_RGB_Image[0]);
	namedWindow("RESIZED GREEN CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("RESIZED GREEN CHANNEL", Final_RGB_Image[1]);
	namedWindow("RESIZED BLUE CHANNEL", CV_WINDOW_AUTOSIZE);
	imshow("RESIZED BLUE CHANNEL", Final_RGB_Image[2]);

	// Merge three channels to an RGB image
	merge(Final_RGB_Image, 3, Final_Image);
	namedWindow("FINAL DISPLAY WINDOW", CV_WINDOW_AUTOSIZE);
	imshow("FINAL DISPLAY WINDOW", Final_Image);

	//Finally release all OpenCL allocated objects and host buffers
	clStatus = clReleaseKernel(kernel_magnify_RED);
	clStatus = clReleaseKernel(kernel_magnify_GREEN);
	clStatus = clReleaseKernel(kernel_magnify_BLUE);
	clStatus = clReleaseProgram(program_magnify);
	clStatus = clReleaseMemObject(Final_RED_clmem);
	clStatus = clReleaseMemObject(Final_GREEN_clmem);
	clStatus = clReleaseMemObject(Final_BLUE_clmem);
	clStatus = clReleaseMemObject(RED_clmem);
	clStatus = clReleaseMemObject(GREEN_clmem);
	clStatus = clReleaseMemObject(BLUE_clmem);
	clStatus = clReleaseCommandQueue(command_queue_magnify);
	clStatus = clReleaseContext(context);
	free(Final_RED_Channel);
	free(Final_GREEN_Channel);
	free(Final_BLUE_Channel);
	free(RED_Channel);
	free(GREEN_Channel);
	free(BLUE_Channel);
	Final_Image.release();
	Final_RGB_Image->release();
	RGB_Image->release();
	Image.release();

	waitKey(0);
	return 0;
}
