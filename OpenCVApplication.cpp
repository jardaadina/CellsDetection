// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#include <numeric>

using namespace std;

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
void showHistogram(const std::string& name, double* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	double max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255));
	}
	imshow(name, imgHist);
}
void additiveFactor()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		int factor;
		cout << "Introduce factor: ";
		cin >> factor;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int newVal = (int)val + factor;
				if (newVal < 0)
					newVal = 0;
				if (newVal > 255)
					newVal = 255;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("additive image", dst);
		waitKey();
	}
}
void multiplicativeFactor()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		int factor;
		cout << "Introduce factor: ";
		cin >> factor;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				float newVal = (float)val * factor;
				if (newVal < 0)
					newVal = 0;
				if (newVal > 255)
					newVal = 255;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("multiplicative image", dst);
		waitKey();
	}
}
void squaredImage()
{
	int height = 256;
	int width = 256;
	Mat myImage = Mat(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b val;
			if (i < 128 && j < 128)
				val = Vec3b(255, 255, 255);
			else if (i < 128 && j >= 128)
				val = Vec3b(0, 0, 255);
			else if (i >= 128 && j < 128)
				val = Vec3b(0, 255, 0);
			else
				val = Vec3b(0, 255, 255);
			myImage.at<Vec3b>(i, j) = val;
		}
	}
	imshow("result", myImage);
	waitKey();
}
void inverseMatrix()
{
	int height = 3;
	int width = 3;
	Mat myMat = Mat(height, width, CV_32FC1);
	cout << "Introduce values of the matrix: ";
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float val;
			cin >> val;
			myMat.at<float>(i, j) = val;
		}
	}
	Mat newMat = myMat.inv();
	cout << newMat;
	getchar();
	getchar();
}
void splitImageByRGB()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat blue = Mat(height, width, CV_8UC1);
		Mat green = Mat(height, width, CV_8UC1);
		Mat red = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				blue.at<uchar>(i, j) = val[0];
				green.at<uchar>(i, j) = val[1];
				red.at<uchar>(i, j) = val[2];
			}
		}
		imshow("input image", src);
		imshow("blue image", blue);
		imshow("green image", green);
		imshow("red image", red);
		waitKey();
	}
}
void colorToGrayScale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				uchar avg = (val[0] + val[1] + val[2]) / 3.0;
				dst.at<uchar>(i, j) = avg;
			}
		}
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}
void binarizeImage()
{
	char fname[MAX_PATH];
	int treshold;
	cin >> treshold;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val < treshold)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
			}
		}
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}
void convertRGBtoHSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat matH = Mat(height, width, CV_8UC1);
		Mat matS = Mat(height, width, CV_8UC1);
		Mat matV = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				float red = (float)val[2] / 255.0;
				float green = (float)val[1] / 255.0;
				float blue = (float)val[0] / 255.0;
				float M = max(red, max(green, blue));
				float m = min(red, min(green, blue));
				float C = M - m;
				float H, S, V;
				V = M;
				if (V != 0)
					S = C / V;
				else
					S = 0;
				if (C != 0)
				{
					if (M == red)
						H = 60 * (green - blue) / C;
					if (M == green)
						H = 120 + 60 * (blue - red) / C;
					if (M == blue)
						H = 240 + 60 * (red - green) / C;
				}
				else
				{
					H = 0;
					if (H < 0)
						H = H + 360;
				}
				uchar H_norm = H / 360 * 255;
				uchar S_norm = S * 255;
				uchar V_norm = V * 255;
				matH.at<uchar>(i, j) = H_norm;
				matS.at<uchar>(i, j) = S_norm;
				matV.at<uchar>(i, j) = V_norm;
			}
		}
		imshow("input image", src);
		imshow("H", matH);
		imshow("S", matS);
		imshow("V", matV);
		waitKey();
	}
}
bool isInside(Mat& img, int i, int j)
{
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	return false;
}
void verifyIsInside()
{
	char fname[MAX_PATH];
	int x, y;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		cout << "Image resolution " << height << " " << width << "\n";
		cout << "Insert x and y\n";
		while (true)
		{
			cin >> x;
			if (x == -1000)
				break;
			cin >> y;
			if (y == -1000)
				break;
			if (isInside(src, x, y) == true)
				cout << "inside\n";
			else
				cout << "not inside\n";
		}
		waitKey();
	}
}
int* hist = (int*)calloc(256, sizeof(int));
double* fdp = (double*)calloc(256, sizeof(double));
void showAllHistograms()
{
	char fname[MAX_PATH];
	int m;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		cout << "Insert m value: ";
		cin >> m;
		if (m < 0 && m>256)
			break;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				hist[src.at < uchar >(i, j)]++;
		for (int i = 0; i < 256; i++)
		{
			fdp[i] = (double)hist[i] / (double)(height * width);
		}
		imshow("Original", src);
		showHistogram("Histogram", hist, 256, 200);
		showHistogram("Histogram with m accumulators", hist, m, 200);
		showHistogram("Normalized Histogram", fdp, 256, 200);
		for (int i = 0; i < 256; i++)
			cout << fdp[i] << " ";
		cout << "\n";
		waitKey();
	}
}
void reduceGrayLevelsPlusFloydSteinberg()
{
	char fname[MAX_PATH];
	int m;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat dstAux = Mat(height, width, CV_32FC1);
		Mat dstCorectat = Mat(height, width, CV_8UC1);
		memset(hist, 0, 256 * sizeof(int));
		memset(fdp, 0, 256 * sizeof(double));
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				hist[src.at < uchar >(i, j)]++;
				dstAux.at<float>(i, j) = (float)src.at<uchar>(i, j);
			}
		for (int i = 0; i < 256; i++)
		{
			fdp[i] = (double)hist[i] / (double)(height * width);
		}
		int w = 5;
		float th = 0.0003;
		int nrPraguri = 2;
		vector<int>praguri;
		praguri.push_back(0);
		for (int i = w; i < 255 - w; i++)
		{
			float v = 0;
			int ok = 1;
			for (int k = i - w; k <= i + w; k++)
			{
				if (fdp[i] < fdp[k])
				{
					ok = 0;
				}
				v += fdp[k];
			}
			v /= (2 * w + 1);
			if (ok && fdp[i] > v + th)
			{
				praguri.push_back(i);
				nrPraguri++;
			}
		}
		praguri.push_back(255);
		for (int i = 0; i < nrPraguri; i++)
			cout << praguri[i] << " ";
		int mini = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				mini = abs(src.at<uchar>(i, j) - praguri[0]);
				int prag = 0;
				for (int k = 1; k < nrPraguri; k++)
				{
					if (mini > abs(src.at<uchar>(i, j) - praguri[k]))
					{
						mini = abs(src.at<uchar>(i, j) - praguri[k]);
						prag = praguri[k];
					}
				}
				dst.at<uchar>(i, j) = prag;
			}
		}
		int* reducedHist = (int*)calloc(256, sizeof(int));
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				reducedHist[dst.at < uchar >(i, j)]++;
		float newMini = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				newMini = abs(dstAux.at<float>(i, j) - praguri[0]);
				int prag = 0;
				for (int k = 1; k < nrPraguri; k++)
				{
					if (newMini > abs(dstAux.at<float>(i, j) - praguri[k]))
					{
						newMini = abs(dstAux.at<float>(i, j) - praguri[k]);
						prag = praguri[k];
					}
				}
				dstCorectat.at<uchar>(i, j) = prag;
				float eroare = dstAux.at<float>(i, j) - prag;
				if (isInside(dstAux, i, j + 1))
					dstAux.at<float>(i, j + 1) = dstAux.at<float>(i, j + 1) + 7 * eroare / 16;
				if (isInside(dstAux, i + 1, j - 1))
					dstAux.at<float>(i + 1, j - 1) = dstAux.at<float>(i + 1, j - 1) + 3 * eroare / 16;
				if (isInside(dstAux, i + 1, j))
					dstAux.at<float>(i + 1, j) = dstAux.at<float>(i + 1, j) + 5 * eroare / 16;
				if (isInside(dstAux, i + 1, j + 1))
					dstAux.at<float>(i + 1, j + 1) = dstAux.at<float >(i + 1, j + 1) + eroare / 16;
			}
		}
		int* correctedHist = (int*)calloc(256, sizeof(int));
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				correctedHist[dstCorectat.at < uchar >(i, j)]++;
		imshow("Original", src);
		imshow("Destinatie", dst);
		imshow("Corectie aplicata", dstCorectat);
		showHistogram("Original Histogram", hist, 256, 200);
		showHistogram("Reduced Histogram", reducedHist, 256, 200);
		showHistogram("Corrected Histogram", correctedHist, 256, 200);
		waitKey();
	}
}
void printGeometricFeatures(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat src = *((Mat*)param);
		if (src.at<Vec3b>(y, x) == Vec3b(255, 255, 255)) return;// alb=fundal
		Vec3b color = src.at<Vec3b>(y, x);
		Vec3b newColor = Vec3b(255, 255, 0);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();
		int aria = 0, ariax = 0, ariay = 0;
		float centrux = 0, centruy = 0;
		//pentru axa de alungire
		float termen1, termen2, termen3;
		termen1 = termen2 = termen3 = 0;
		float axaAlungire = 0;
		float perimetru = 0;
		float factorSubtiere, factorAspect;
		factorSubtiere = factorAspect = 0;
		int cmax, cmin, rmax, rmin;
		cmax = rmax = INT_MIN;
		cmin = rmin = INT_MAX;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<Vec3b>(i, j) == color)
				{
					aria++;
					ariax += i;
					ariay += j;
					int flag = 0;
					int dx[8] = { 0, 0, 1, 1, 1, -1, -1, -1 };
					int dy[8] = { 1, -1, 0, 1, -1, 0, 1, -1 };
					for (int k = 0; k < 8; k++)
					{
						int inou = i + dx[k];
						int jnou = j + dy[k];
						if (isInside(src, inou, jnou))
						{
							if (src.at<Vec3b>(inou, jnou) != color)
							{
								if (!flag) {
									perimetru++;
									flag = 1;
								}
								dst.at<Vec3b>(inou, jnou) = newColor;
							}
						}
					}
					cmax = max(cmax, j);
					rmax = max(rmax, i);
					cmin = min(cmin, j);
					rmin = min(rmin, i);
				}
			}
		}
		centrux = (float)ariax / (float)aria;
		centruy = (float)ariay / (float)aria;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<Vec3b>(i, j) == color)
				{
					termen1 += (i - centrux) * (j - centruy);
					termen2 += (j - centruy) * (j - centruy);
					termen3 += (i - centrux) * (i - centrux);
				}
			}
		}
		axaAlungire = 0.5 * atan2(2 * termen1, (termen2 - termen3)) * 180 / PI;
		perimetru = cvRound(perimetru * PI / 4.0 * 100) / 100.0;
		factorSubtiere = 4 * PI * (float)aria / (float)(perimetru * perimetru);
		factorAspect = (float)(cmax - cmin + 1) / (float)(rmax - rmin + 1);
		//desenam centrul de masa
		for (int x = centrux + 5; x >= centrux - 5; x--)
			if (isInside(src, x, centruy))
				dst.at<Vec3b>(x, centruy) = newColor;
		for (int y = centruy + 5; y >= centruy - 5; y--)
			if (isInside(src, centrux, y))
				dst.at<Vec3b>(centrux, y) = newColor;
		//desenam axa de alungire
		if (termen1 == 0) {
			axaAlungire = (termen2 > termen3) ? 0 : 90;
		}
		else {
			axaAlungire = 0.5 * atan2(2 * termen1, (termen2 - termen3)) * 180 / CV_PI;
		}
		float rad = axaAlungire * CV_PI / 180.0;
		Point p1(centruy, centrux);
		Point p2;
		if (fabs(axaAlungire - 90) < 5)
			p2 = Point(centruy, centrux + 30);
		else if (fabs(axaAlungire) < 5 || fabs(axaAlungire - 180) < 5)
			p2 = Point(centruy + 30, centrux);
		else
			p2 = Point(centruy + 30 * cos(rad), centrux + 30 * sin(rad));
		if (isInside(dst, p1.y, p1.x) && isInside(dst, p2.y, p2.x))
			line(dst, p1, p2, Scalar(255, 255, 0), 2);
		Mat p = Mat::zeros(height, width, CV_8UC3);
		vector<int>hp(height, 0);
		vector<int>vp(width, 0);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<Vec3b>(i, j) == color)
				{
					hp[i]++;
					vp[j]++;
				}
			}
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < hp[i]; j++)
				p.at<Vec3b>(i, j) = Vec3b(128, 0, 128);
		}
		for (int j = 0; j < width; j++)
		{
			for (int i = height - vp[j]; i < height; i++)
				p.at<Vec3b>(i, j) = Vec3b(128, 0, 128);
		}
		cout << "Aria este egala cu " << aria << "\n";
		cout << "Centrul de masa are coordonatele ";
		printf("%.2f %.2f\n", centruy, centrux);
		cout << "Axa de alungire este egala cu ";
		printf("%.2f\n", axaAlungire);
		cout << "Perimetrul obiectului este egal cu " << perimetru << "\n";
		cout << "Factorul de subtiere este egal cu ";
		printf("%.2f\n", factorSubtiere);
		cout << "Factorul de aspect este egal cu ";
		printf("%.2f\n", factorAspect);
		imshow("Geometric properties", dst);
		imshow("Projections", p);
	}
}
void computeGeometricFeatures()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		imshow("Source image", src);
		setMouseCallback("Source image", printGeometricFeatures, &src);
		waitKey();
		destroyAllWindows();
	}
}
void BFSN8()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
		int label = 0;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		default_random_engine gen;
		uniform_int_distribution<int>d(0, 255);
		Vec3b color[1000];
		for (int i = 0; i < 1000; i++)
			color[i] = Vec3b(d(gen), d(gen), d(gen));
		int dx[8] = { 0, 0, 1, 1, 1, -1, -1, -1 };
		int dy[8] = { 1, -1, 0, 1, -1, 0, 1, -1 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					label++;
					queue<pair<int, int>>Q;
					labels.at<int>(i, j) = label;
					dst.at<Vec3b>(i, j) = color[label];
					Q.push({ i,j });
					while (!Q.empty())
					{
						pair<int, int> q = Q.front();
						Q.pop();
						int qi = q.first;
						int qj = q.second;
						for (int k = 0; k < 8; k++)
						{
							int inou = qi + dx[k];
							int jnou = qj + dy[k];
							if (isInside(src, inou, jnou))
							{
								if (src.at<uchar>(inou, jnou) == 0 && labels.at<int>(inou, jnou) == 0)
								{
									labels.at<int>(inou, jnou) = label;
									dst.at<Vec3b>(inou, jnou) = color[label];
									Q.push({ inou,jnou });
								}
							}
						}
					}
				}
			}
		}
		printf("The number of objects is %d\n", label);
		imshow("Source image", src);
		imshow("Labeled N8 image", dst);
		waitKey();
	}
}
void BFSN4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
		int label = 0;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		int dx[4] = { -1,0,1,0 };
		int dy[4] = { 0,-1,0,1 };
		default_random_engine gen;
		uniform_int_distribution<int>d(0, 255);
		Vec3b color[1000];
		for (int i = 0; i < 1000; i++)
			color[i] = Vec3b(d(gen), d(gen), d(gen));
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					label++;
					queue<pair<int, int>>Q;
					labels.at<int>(i, j) = label;
					dst.at<Vec3b>(i, j) = color[label];
					Q.push({ i,j });
					while (!Q.empty())
					{
						pair<int, int> q = Q.front();
						Q.pop();
						int qi = q.first;
						int qj = q.second;
						for (int k = 0; k < 4; k++)
						{
							int inou = qi + dx[k];
							int jnou = qj + dy[k];
							if (isInside(src, inou, jnou))
							{
								if (src.at<uchar>(inou, jnou) == 0 && labels.at<int>(inou, jnou) == 0)
								{
									labels.at<int>(inou, jnou) = label;
									dst.at<Vec3b>(inou, jnou) = color[label];
									Q.push({ inou,jnou });
								}
							}
						}
					}
				}
			}
		}
		printf("The number of objects is %d\n", label);
		imshow("Source image", src);
		imshow("Labeled N4 image", dst);
		waitKey();
	}
}
void twoStepsProcessing()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst1 = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
		Mat dst2 = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
		int label = 0;
		vector<int> parent(1000, 0);
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		vector<vector<int>>edges(1000);
		default_random_engine gen;
		uniform_int_distribution<int>d(0, 255);
		Vec3b color[1000];
		for (int i = 0; i < 1000; i++)
			color[i] = Vec3b(d(gen), d(gen), d(gen));
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					vector<int>L;
					int dx[8] = { 0, 0, 1, 1, 1, -1, -1, -1 };
					int dy[8] = { 1, -1, 0, 1, -1, 0, 1, -1 };
					for (int k = 0; k < 8; k++)
					{
						int inou = i + dx[k];
						int jnou = j + dy[k];
						if (isInside(src, inou, jnou))
						{
							if (labels.at<int>(inou, jnou) > 0)
								L.push_back(labels.at<int>(inou, jnou));
						}

					}
					if (L.size() == 0)
					{
						label++;
						labels.at<int>(i, j) = label;
						parent[label] = label;
					}
					else
					{
						int x = *(min_element(L.begin(), L.end()));
						labels.at<int>(i, j) = x;
						for (int k = 0; k < L.size(); k++)
						{
							int y = L[k];
							if (y != x)
							{
								edges[x].push_back(y);
								edges[y].push_back(x);
								parent[y] = x;
							}
						}
						dst1.at<Vec3b>(i, j) = color[labels.at<int>(i, j) % 1000];
					}
				}
			}
		}
		auto findRoot = [&](int x)
			{
				while (parent[x] != x)
					x = parent[x];
				return x;
			};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) > 0)
				{
					labels.at<int>(i, j) = findRoot(labels.at<int>(i, j));
					dst2.at<Vec3b>(i, j) = color[labels.at<int>(i, j) % 1000];
				}
			}
		}
		printf("The number of objects is %d\n", label);
		imshow("Source Image", src);
		imshow("1/2 Image", dst1);
		imshow("2/2 Image", dst2);
		waitKey();
	}
}
void showOutline()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.at<Vec3b>(i, j) = Vec3b(src.at<uchar>(i, j), src.at<uchar>(i, j), src.at<uchar>(i, j));
		vector<int> cod;
		vector<int> derivata;
		vector<pair<int, int>> contur;
		bool found = false;
		int dir;
		for (int i = 0; i < height && !found; i++)
		{
			for (int j = 0; j < width && !found; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					found = true;
					//incepem conturul
					contur.push_back({ i,j });
					dst.at<Vec3b>(i, j) = Vec3b(255, 0, 255);
					dir = 7;
					int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
					int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
					int inou, jnou;
					int x = i, y = j;
					while (true)
					{
						int startDir = (dir + 6) % 8;
						bool ok = false;
						for (int k = 0; k < 8; k++)
						{
							int currDir = (startDir + k) % 8;
							inou = x + dx[currDir];
							jnou = y + dy[currDir];
							if (isInside(src, inou, jnou) && src.at<uchar>(inou, jnou) == 0)
							{
								x = inou;
								y = jnou;
								contur.push_back({ inou,jnou });
								cod.push_back(currDir);
								if (cod.size() > 1)
									derivata.push_back((cod[cod.size() - 1] - cod[cod.size() - 2] + 8) % 8);
								dst.at<Vec3b>(inou, jnou) = Vec3b(255, 0, 255);
								dir = currDir;
								ok = true;
								break;
							}
						}
						if (!ok)
							break;
						if (contur.size() >= 3 && contur[contur.size() - 1] == contur[1] && contur[contur.size() - 2] == contur[0])
						{
							cod.pop_back();
							break;
						}
					}
				}
			}
		}
		imshow("Soruce Image", src);
		imshow("Outlined Image", dst);
		cout << "Codul inlantuit: ";
		for (auto it = cod.begin(); it != cod.end(); it++)
		{
			cout << *it << " ";
		}
		cout << endl;
		cout << "Derivata: ";
		for (auto it = derivata.begin(); it != derivata.end(); it++)
		{
			cout << *it << " ";
		}
		cout << endl;
		waitKey();
	}
}
void reconstructImage()
{
	char fname[MAX_PATH];
	if (!openFolderDlg(fname))
		return;
	char imN[MAX_PATH];
	strcpy(imN, fname);
	strcat(imN, "\\gray_background.bmp");
	char txtN[MAX_PATH];
	strcpy(txtN, fname);
	strcat(txtN, "\\reconstruct.txt");
	Mat dst = imread(imN, IMREAD_COLOR);
	int height = dst.rows;
	int width = dst.cols;
	int xStart, yStart;
	int n, dir;
	FILE* f = fopen(txtN, "r");
	fscanf(f, "%d %d %d", &xStart, &yStart, &n);
	dst.at<Vec3b>(xStart, yStart) = Vec3b(255, 0, 255);
	int x = xStart, y = yStart;
	for (int i = 0; i < n; i++)
	{
		int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		fscanf(f, "%d", &dir);
		x += dx[dir];
		y += dy[dir];
		dst.at<Vec3b>(x, y) = Vec3b(255, 0, 255);
	}
	imshow("Reconstructed Image", dst);
	waitKey();
}
Mat dilateImage(Mat& src, int n)
{
	Mat current = src.clone();
	Mat temp = src.clone();
	int height, width;
	height = src.rows;
	width = src.cols;
	for (int k = 0; k < n; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar mini = 255;
				int dx[9] = { 0, 0, -1, -1, -1, 0, 1, 1, 1 };
				int dy[9] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
				int inou, jnou;
				for (int q = 0; q < 9; q++)
				{
					inou = i + dx[q];
					jnou = j + dy[q];
					if (isInside(current, inou, jnou))
						mini = min(mini, current.at<uchar>(inou, jnou));
				}
				temp.at<uchar>(i, j) = mini;
			}
		}
		current = temp.clone();
	}
	return current;
}
Mat erodeImage(Mat& src, int n)
{
	Mat current = src.clone();
	Mat temp = src.clone();
	int height, width;
	height = src.rows;
	width = src.cols;
	for (int k = 0; k < n; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar maxi = 0;
				int dx[9] = { 0, 0, -1, -1, -1, 0, 1, 1, 1 };
				int dy[9] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
				int inou, jnou;
				for (int q = 0; q < 9; q++)
				{
					inou = i + dx[q];
					jnou = j + dy[q];
					if (isInside(current, inou, jnou))
						maxi = max(maxi, current.at<uchar>(inou, jnou));
				}
				temp.at<uchar>(i, j) = maxi;
			}
		}
		current = temp.clone();
	}
	return current;
}
void morphologyOperations()
{
	char fname[MAX_PATH];
	int n;
	cout << "Introdu N: ";
	cin >> n;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dilated = dilateImage(src, n);
		Mat eroded = erodeImage(src, n);
		Mat openedOneTime = dilateImage(erodeImage(src, 1), 1);
		Mat closedOneTime = erodeImage(dilateImage(src, 1), 1);
		Mat openedNTimes = src.clone();
		for (int i = 0; i < n; i++)
			openedNTimes = dilateImage(erodeImage(openedNTimes, 1), 1);
		Mat closedNTimes = src.clone();
		for (int i = 0; i < n; i++)
			closedNTimes = erodeImage(dilateImage(closedNTimes, 1), 1);
		imshow("Original", src);
		imshow("Dilated Image", dilated);
		imshow("Eroded Image", eroded);
		imshow("Opened Image Once", openedOneTime);
		imshow("Closed Image Once", closedOneTime);
		imshow("Opened Image N", openedNTimes);
		imshow("Closed Image N", closedNTimes);
		waitKey();
	}
}
Mat extractContourMorphologic(Mat& src)
{
	Mat dilated = dilateImage(src, 1);
	Mat contur = Mat::ones(src.size(), CV_8UC1) * 255;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) != dilated.at<uchar>(i, j))
				contur.at<uchar>(i, j) = 0;
	return contur;
}
Mat fillRegionMorphologic(Mat& contur, int x, int y)
{
	Mat filled = Mat::zeros(contur.size(), CV_8UC1);
	filled.at<uchar>(x, y) = 255;
	Mat prev, next;
	prev = next = filled.clone();
	do
	{
		prev = next.clone();
		next = prev.clone();
		for (int i = 0; i < prev.rows; i++)
		{
			for (int j = 0; j < prev.cols; j++)
			{
				if (prev.at<uchar>(i, j) == 255)
				{
					int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
					int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
					int inou, jnou;
					for (int k = 0; k < 8; k++)
					{
						inou = i + dx[k];
						jnou = j + dy[k];
						if (isInside(prev, inou, jnou) && contur.at<uchar>(inou, jnou) == 255 && next.at<uchar>(inou, jnou) == 0)
							next.at<uchar>(inou, jnou) = 255;
					}
				}
			}
		}
	} while (countNonZero(next != prev) > 0);
	Mat rez = Mat::ones(contur.size(), CV_8UC1) * 255;
	for (int i = 0; i < rez.rows; ++i)
		for (int j = 0; j < rez.cols; ++j)
			if (next.at<uchar>(i, j) == 255)
				rez.at<uchar>(i, j) = 0;
	return rez;
}
void fct_umplere(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat* contur = (Mat*)param;
		if (contur->at<uchar>(y, x) == 0)
			return;
		Mat umplere = fillRegionMorphologic(*contur, y, x);
		imshow("Umplere", umplere);
	}
}
void showContourAndFillMorphologic()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat contur = extractContourMorphologic(src);
		imshow("Original", src);
		imshow("Contur", contur);
		setMouseCallback("Contur", fct_umplere, &contur);
		waitKey();
		destroyAllWindows();
	}
}
int* returnHistogram(Mat src)
{
	int* h = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			h[src.at<uchar>(i, j)]++;
	return h;
}
void showHistogramProperties()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int* h = returnHistogram(src);
		imshow("Original Image", src);
		showHistogram("Original", h, 256, 200);
		double media;
		double dim = height * width;
		double sum = 0;
		for (int i = 0; i <= 255; i++)
			sum += i * h[i];
		media = sum / dim;
		cout << "Media: " << media << "\n";
		double fdp[256];
		for (int i = 0; i <= 255; i++)
			fdp[i] = h[i] / dim;
		double deviatie = 0;
		for (int i = 0; i <= 255; i++)
			deviatie += (i - media) * (i - media) * fdp[i];
		deviatie = sqrt(deviatie);
		cout << "Deviatia standard: " << deviatie << "\n";
		int c[256] = { 0 };
		c[0] = h[0];
		for (int i = 1; i < 256; i++)
			c[i] = c[i - 1] + h[i];
		showHistogram("Cumulativa", c, 256, 200);
		waitKey();
	}
}
void binarizeImageBasedOnError()
{
	char fname[MAX_PATH];
	double eroare;
	cout << "Introdu eroarea: ";
	cin >> eroare;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int* h = returnHistogram(src);
		int imax = 255, imin = 0;
		while (h[imin] == 0 && imin < 255)
			imin++;
		while (h[imax] == 0 && imax > 0)
			imax--;
		double t1, t2;
		t1 = (imax + imin) / 2.0;
		t2 = t1;
		while (true) {
			t1 = t2;
			double g1, g2;
			g1 = g2 = 0;
			double n1 = 0, n2 = 0, sum1 = 0, sum2 = 0;
			for (int i = imin; i <= (int)t1; i++)
				n1 += h[i], sum1 += i * h[i];
			for (int i = (int)t1 + 1; i <= imax; i++)
				n2 += h[i], sum2 += i * h[i];
			if (n1 == 0 || n2 == 0) break;
			g1 = sum1 / n1;
			g2 = sum2 / n2;
			t2 = (g1 + g2) / 2.0;
			if (fabs(t2 - t1) < eroare)
				break;
		}
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src.at<uchar>(i, j) < t2)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
		cout << "Pragul de binarizare este: " << t2 << "\n";
		imshow("Binarizat", dst);
		waitKey();
	}
}
void gammaCorrection()
{
	char fname[MAX_PATH];
	int ioutmin, ioutmax, offset;
	double gamma;
	cout << "Introdu IoutMin, IoutMax, Gamma si Offset: \n";
	cin >> ioutmin >> ioutmax >> gamma >> offset;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Original Image", src);
		int* h = returnHistogram(src);
		showHistogram("Original", h, 256, 200);
		Mat neg = src.clone();
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				neg.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		imshow("Negativ Image", neg);
		int hneg[256] = { 0 };
		for (int i = 0; i <= 255; i++)
			hneg[i] = h[255 - i];
		showHistogram("Negativ", hneg, 256, 200);
		int imax = 255, imin = 0;
		while (h[imin] == 0 && imin < 255)
			imin++;
		while (h[imax] == 0 && imax > 0)
			imax--;
		Mat c = src.clone();
		int hcont[256] = { 0 };
		int num = imax - imin;
		if (num == 0) num = 1;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				int pixel = src.at<uchar>(i, j);
				double val = ioutmin + ((double)(pixel - imin) * (ioutmax - ioutmin)) / num;
				if (val < 0) val = 0;
				if (val > 255) val = 255;
				c.at<uchar>(i, j) = (uchar)(val);
				hcont[(int)(val)]++;
			}
		imshow("Contrast Image", c);
		showHistogram("Contrast", hcont, 256, 200);
		Mat g = src.clone();
		int hgamma[256] = { 0 };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				g.at<uchar>(i, j) = 255 * pow((src.at<uchar>(i, j) / 255.0), gamma);
				hgamma[g.at<uchar>(i, j)]++;
			}
		imshow("Gamma Image", g);
		showHistogram("Gamma", hgamma, 256, 200);
		Mat off = src.clone();
		int hoff[256] = { 0 };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				int newVal = src.at<uchar>(i, j) + offset;
				if (newVal > 255)
					newVal = 255;
				off.at<uchar>(i, j) = newVal;
				hoff[off.at<uchar>(i, j)]++;
			}
		imshow("Offset Image", off);
		showHistogram("Offset", hoff, 256, 200);
		waitKey();
	}
}
void equalizeHistogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int h[256] = { 0 };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				h[src.at<uchar>(i, j)]++;
		double fdpc[256] = { 0.0 };
		double sum = 0;
		for (int i = 0; i <= 255; i++)
		{
			sum += h[i];
			fdpc[i] = sum / (height * width);
		}
		Mat dst = src.clone();
		int heq[256] = { 0 };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = 255 * fdpc[src.at<uchar>(i, j)];
				heq[dst.at<uchar>(i, j)]++;
			}
		imshow("Equalized Imae", dst);
		showHistogram("Equalized", heq, 256, 200);
		waitKey();
	}
}
void applyFilters()
{
	//char fname[MAX_PATH];
	//while (openFileDlg(fname))
	//{
	Mat src = imread("C:/Users/Noemi/Desktop/pi/laboratoare/OpenCVApplication-VS2022_OCV490_basic/Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int height = src.rows;
	int width = src.cols;
	int w = 3, k = 1;
	Mat arit = Mat(w, w, CV_32F, Scalar(1.0 / 9.0));
	float dataGauss[9] = { 1,2,1,2,4,2,1,2,1 };
	Mat gauss = Mat(w, w, CV_32F, dataGauss);
	gauss /= 16.0;
	float dataLap[9] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
	Mat lap = Mat(w, w, CV_32F, dataLap);
	float dataHigh[9] = { -1,-1,-1,-1,9,-1,-1,-1,-1 };
	Mat high = Mat(w, w, CV_32F, dataHigh);
	Mat dstArit = Mat(height, width, CV_8UC1);
	Mat dstGauss = Mat(height, width, CV_8UC1);
	Mat dstLap = Mat(height, width, CV_8UC1);
	Mat dstLapFloat(height, width, CV_32F, Scalar(0));
	Mat dstHigh = Mat(height, width, CV_8UC1);
	Mat dstHighFloat = Mat(height, width, CV_32F, Scalar(0));
	for (int i = k; i < height - k; i++)
		for (int j = k; j < width - k; j++)
		{
			float valArit, valGauss, valLap, valHigh;
			valArit = valGauss = valLap = valHigh = 0;
			for (int u = 0; u < w; u++)
				for (int v = 0; v < w; v++)
				{
					valArit += arit.at<float>(u, v) * src.at<uchar>(i + u - k, j + v - k);
					valGauss += gauss.at<float>(u, v) * src.at<uchar>(i + u - k, j + v - k);
					valLap += lap.at<float>(u, v) * src.at<uchar>(i + u - k, j + v - k);
					valHigh += high.at<float>(u, v) * src.at<uchar>(i + u - k, j + v - k);
				}
			dstArit.at<uchar>(i, j) = (uchar)valArit;
			dstGauss.at<uchar>(i, j) = (uchar)valGauss;
			dstLapFloat.at<float>(i, j) = valLap;
			dstHighFloat.at<float>(i, j) = valHigh;
		}
	normalize(dstLapFloat, dstLap, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(dstHighFloat, dstHigh, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Original", src);
	imshow("FTJ Medie", dstArit);
	imshow("FTJ Gaussian", dstGauss);
	imshow("FTS Laplace", dstLap);
	imshow("FTS High Pass", dstHigh);
	waitKey();
	//}
}
void centering_transform(Mat img)
{
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
}
void generic_frequency_domain_filter()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Original", src);
		int height = src.rows;
		int width = src.cols;
		int r, a;
		r = a = 20;
		Mat srcf;
		//ne trebuie imagine cu valori float
		src.convertTo(srcf, CV_32FC1);
		centering_transform(srcf);
		//aplicare fourier, se obtine imagine cu numere complexe
		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
		//impartim in parte reala si parte imaginara
		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);//channels[0] - e partea reala
		//channels[1] - e partea imaginara
//calculam magnitudinea, faza si phi
		Mat mag, phi;
		magnitude(channels[0], channels[1], mag);
		phase(channels[0], channels[1], phi);
		mag += 1;
		log(mag, mag);
		Mat log_img;
		normalize(mag, log_img, 0, 255, NORM_MINMAX);
		log_img.convertTo(log_img, CV_8UC1);
		imshow("Log spectrum", log_img);
		merge(channels, 2, fourier);
		//aplicare fourier inversa
		Mat dst, dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		dstf.convertTo(dst, CV_8UC1);
		imshow("Image after IDFT", dst);
		//operatii de filtrarea aplicate pe coeficientii fourieri
		//Filtru Trece Jos
		Mat channels_aux1[]{ channels[0].clone(),channels[1].clone() };
		for (int u = 0; u < mag.rows; u++)
			for (int v = 0; v < mag.cols; v++)
			{
				float val = (height / 2.0 - u) * (height / 2.0 - u) + (width / 2.0 - v) * (width / 2.0 - v);
				if (val > r * r)
				{
					channels_aux1[0].at<float>(u, v) = 0;
					channels_aux1[1].at<float>(u, v) = 0;
				}
			}
		merge(channels_aux1, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX);
		dst.convertTo(dst, CV_8UC1);
		imshow("Ideal low pass filter", dst);
		//Filtru Trece Sus
		Mat channels_aux2[]{ channels[0].clone(),channels[1].clone() };
		for (int u = 0; u < mag.rows; u++)
			for (int v = 0; v < mag.cols; v++)
			{
				float val = (height / 2.0 - u) * (height / 2.0 - u) + (width / 2.0 - v) * (width / 2.0 - v);
				if (val <= r * r)
				{
					channels_aux2[0].at<float>(u, v) = 0;
					channels_aux2[1].at<float>(u, v) = 0;
				}
			}
		merge(channels_aux2, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX);
		dst.convertTo(dst, CV_8UC1);
		imshow("Ideal high pass filter", dst);
		//Filtru Gaussian Trece Jos
		Mat channels_aux3[]{ channels[0].clone(),channels[1].clone() };
		for (int u = 0; u < mag.rows; u++)
			for (int v = 0; v < mag.cols; v++)
			{
				float val = ((height / 2.0 - u) * (height / 2.0 - u) + (width / 2.0 - v) * (width / 2.0 - v)) / (a * a);
				channels_aux3[0].at<float>(u, v) = channels_aux3[0].at<float>(u, v) * exp(-val);
				channels_aux3[1].at<float>(u, v) = channels_aux3[1].at<float>(u, v) * exp(-val);
			}
		merge(channels_aux3, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX);
		dst.convertTo(dst, CV_8UC1);
		imshow("Ideal gaussian low pass filter", dst);
		//Filtru Gaussian Trece Sus
		Mat channels_aux4[]{ channels[0].clone(),channels[1].clone() };
		for (int u = 0; u < mag.rows; u++)
			for (int v = 0; v < mag.cols; v++)
			{
				float val = ((height / 2.0 - u) * (height / 2.0 - u) + (width / 2.0 - v) * (width / 2.0 - v)) / (a * a);
				channels_aux4[0].at<float>(u, v) = channels_aux4[0].at<float>(u, v) * (1 - exp(-val));
				channels_aux4[1].at<float>(u, v) = channels_aux4[1].at<float>(u, v) * (1 - exp(-val));
			}
		merge(channels_aux4, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX);
		dst.convertTo(dst, CV_8UC1);
		imshow("Ideal gaussian high pass filter", dst);
		waitKey();
	}
}
void medianNoiseFiltering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w, k;
		cout << "Introduceti dimensiunea ferestrei:";
		cin >> w;
		k = (w - 1) / 2;
		//Mat f = Mat(w, w, CV_8UC1);
		Mat dst = src.clone();
		double t = (double)getTickCount();
		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{
				vector<uchar>v;
				for (int u = -k; u <= k; u++)
					for (int q = -k; q <= k; q++)
						v.push_back(src.at<uchar>(i + u, j + q));
				sort(v.begin(), v.end());
				dst.at<uchar>(i, j) = v[w * w / 2];//atribuim mediana

			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("Original", src);
		imshow("Median filtering", dst);
		waitKey();
	}
}
void gaussianNoiseFiltering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w, k;
		cout << "Introduceti dimensiunea ferestrei:";
		cin >> w;
		k = (w - 1) / 2;
		int x0, y0;
		x0 = y0 = w / 2;
		float sigma = w / 6.0;
		Mat g = Mat(w, w, CV_32F);
		Mat dst1 = src.clone();
		Mat dst2 = src.clone();
		double t = (double)getTickCount();
		//Filtering bidimensional
		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++)
				g.at<float>(i, j) = 1 / (2 * PI * sigma * sigma) * exp(-(pow(i - x0, 2) + pow(j - y0, 2)) / (2 * pow(sigma, 2)));
		g /= sum(g)[0];
		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{
				float newVal = 0;
				for (int u = 0; u < w; u++)
					for (int v = 0; v < w; v++)
						newVal += (float)src.at<uchar>(i + u - k, j + v - k) * g.at<float>(u, v);
				dst1.at<uchar>(i, j) = (uchar)newVal;
			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time for 2D = %.3f [ms]\n", t * 1000);
		t = (double)getTickCount();
		//Filtering unidimensional
		vector<float>gx(w);
		vector<float>gy(w);
		for (int i = 0; i < w; i++)
		{
			gx[i] = 1.0 / (sqrt(2 * PI) * sigma) * exp(-pow(i - x0, 2) / (2 * pow(sigma, 2)));
			gy[i] = 1.0 / (sqrt(2 * PI) * sigma) * exp(-pow(i - y0, 2) / (2 * pow(sigma, 2)));
		}
		float normx = accumulate(gx.begin(), gx.end(), 0.0f);
		float normy = accumulate(gy.begin(), gy.end(), 0.0f);
		for (auto& val : gx)
			val /= normx;
		for (auto& val : gy)
			val /= normy;
		Mat aux = src.clone();
		aux.convertTo(aux, CV_32F);
		for (int i = k; i < height - k; i++)
			for (int j = k; j < width - k; j++)
			{
				float newVal = 0;
				for (int u = -k; u <= k; u++)
					newVal += (float)src.at<uchar>(i, j + u) * gx[u + k];
				aux.at<float>(i, j) = newVal;
			}
		for (int i = k; i < height - k; i++)
			for (int j = k; j < width - k; j++)
			{
				float newVal = 0;
				for (int u = -k; u <= k; u++)
					newVal += aux.at<float>(i + u, j) * gy[u + k];
				dst2.at<uchar>(i, j) = (uchar)newVal;
			}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time 1D = %.3f [ms]\n", t * 1000);
		imshow("Original", src);
		imshow("2D Gaussian filter", dst1);
		imshow("1D Gaussian filter", dst2);
		waitKey();
	}
}
void prewittNucleusApplication()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int prewittDataX[9] = { -1,0,1,-1,0,1,-1,0,1 };
		int prewittDataY[9] = { 1,1,1,0,0,0,-1,-1,-1 };
		int w, k;
		k = 1;
		w = 2 * k + 1;
		Mat pwx = Mat(w, w, CV_32SC1, prewittDataX);
		Mat pwy = Mat(w, w, CV_32SC1, prewittDataY);
		Mat dst1 = Mat::zeros(src.size(), CV_32SC1);
		Mat dst2 = Mat::zeros(src.size(), CV_32SC1);
		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{
				int newValX = 0;
				int newValY = 0;
				for (int u = 0; u < w; u++)
					for (int v = 0; v < w; v++)
					{
						newValX += pwx.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
						newValY += pwy.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
					}
				dst1.at<int>(i, j) = newValX;
				dst2.at<int>(i, j) = newValY;
			}
		}
		Mat phase = Mat::zeros(src.size(), CV_32FC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				int a, b;
				a = dst2.at<int>(i, j);
				b = dst1.at<int>(i, j);
				phase.at<float>(i, j) = atan2((float)a, (float)b);
			}
		phase += CV_PI;
		phase /= (2 * CV_PI);
		phase *= 255;
		Mat mag = Mat::zeros(src.size(), CV_32FC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				int a, b;
				a = dst1.at<int>(i, j);
				b = dst2.at<int>(i, j);
				mag.at<float>(i, j) = sqrt((float)(a * a + b * b));
			}
		mag /= (sqrt(2) * 3);
		Mat dst1_norm, dst2_norm, phase_norm, mag_norm;
		normalize(dst1, dst1_norm, 0, 255, NORM_MINMAX, CV_8UC1);
		normalize(dst2, dst2_norm, 0, 255, NORM_MINMAX, CV_8UC1);
		phase.convertTo(phase_norm, CV_8UC1);
		mag.convertTo(mag_norm, CV_8UC1);
		int th;
		cout << "Introduceti pragul:";
		cin >> th;
		Mat bin = Mat::zeros(src.size(), CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (mag_norm.at<uchar>(i, j) < th)
					bin.at<uchar>(i, j) = 0;
				else
					bin.at<uchar>(i, j) = 255;
		imshow("Original", src);
		imshow("PrewittX", dst1_norm);
		imshow("PrewittY", dst2_norm);
		imshow("Prewitt(phase)", phase_norm);
		imshow("Prewitt(magnitude)", mag_norm);
		imshow("Prewitt(binarized)", bin);
		waitKey();
	}
}
void gaussianFilterAndSobelNucleusApplication()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w, k;
		k = 1;
		w = 2 * k + 1;
		float dataGauss[9] = { 1,2,1,2,4,2,1,2,1 };
		Mat gauss = Mat(w, w, CV_32F, dataGauss);
		gauss /= 16.0;
		Mat dstGauss = Mat(height, width, CV_8UC1);
		for (int i = k; i < height - k; i++)
			for (int j = k; j < width - k; j++)
			{
				float valGauss;
				valGauss = 0;
				for (int u = 0; u < w; u++)
					for (int v = 0; v < w; v++)
						valGauss += gauss.at<float>(u, v) * src.at<uchar>(i + u - k, j + v - k);
				dstGauss.at<uchar>(i, j) = (uchar)valGauss;
			}
		int sobelDataX[9] = { -1,0,1,-2,0,2,-1,0,1 };
		int sobelDataY[9] = { 1,2,1,0,0,0,-1,-2,-1 };
		Mat sbx = Mat(w, w, CV_32SC1, sobelDataX);
		Mat sby = Mat(w, w, CV_32SC1, sobelDataY);
		Mat dst1 = Mat::zeros(src.size(), CV_32SC1);
		Mat dst2 = Mat::zeros(src.size(), CV_32SC1);
		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{
				int newValX = 0;
				int newValY = 0;
				for (int u = 0; u < w; u++)
					for (int v = 0; v < w; v++)
					{
						newValX += sbx.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
						newValY += sby.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
					}
				dst1.at<int>(i, j) = newValX;
				dst2.at<int>(i, j) = newValY;
			}
		}
		Mat phase = Mat::zeros(src.size(), CV_32FC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				int a, b;
				a = dst2.at<int>(i, j);
				b = dst1.at<int>(i, j);
				phase.at<float>(i, j) = atan2((float)a, (float)b);
			}
		Mat phase_orig = phase.clone();
		phase += CV_PI;
		phase /= (2 * CV_PI);
		phase *= 255;
		Mat mag = Mat::zeros(src.size(), CV_32FC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				int a, b;
				a = dst1.at<int>(i, j);
				b = dst2.at<int>(i, j);
				mag.at<float>(i, j) = sqrt((float)(a * a + b * b));
			}
		mag /= (sqrt(2) * 4);
		Mat mag_norm;
		mag.convertTo(mag_norm, CV_8UC1);
		Mat mag_max = mag.clone();
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				float angle = phase_orig.at<float>(i, j) * 180.0f / (float)CV_PI;
				if (angle < 0)
					angle += 360.0f;
				int zone = 0;
				if ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5))
					zone = 0;
				else if ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5))
					zone = 1;
				else if ((angle >= 337.5 || angle < 22.5) || (angle >= 157.5 && angle < 202.5))
					zone = 2;
				else if ((angle >= 112.5 && angle < 157.5) || (angle >= 292.5 && angle < 337.5))
					zone = 3;
				float curr = mag.at<float>(i, j);
				float vecin1, vecin2;
				vecin1 = vecin2 = 0;
				if (zone == 0)
				{
					vecin1 = mag.at<float>(i - 1, j);
					vecin2 = mag.at<float>(i + 1, j);
				}
				else if (zone == 1)
				{
					vecin1 = mag.at<float>(i - 1, j + 1);
					vecin2 = mag.at<float>(i + 1, j - 1);
				}
				else if (zone == 2)
				{
					vecin1 = mag.at<float>(i, j - 1);
					vecin2 = mag.at<float>(i, j + 1);
				}
				else if (zone == 3)
				{
					vecin1 = mag.at<float>(i - 1, j - 1);
					vecin2 = mag.at<float>(i + 1, j + 1);
				}
				if (curr >= vecin1 && curr >= vecin2)
					mag_max.at<float>(i, j) = curr;
				else
					mag_max.at<float>(i, j) = 0;
			}
		}
		Mat mag_max_norm;
		mag_max.convertTo(mag_max_norm, CV_8UC1);
		int hist[256] = { 0 };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				hist[mag_max_norm.at<uchar>(i, j)]++;
		float P, K;
		P = 0.1f;
		K = 0.4f;
		int nr_non_muchie = (int)((1.0f - P) * (width * height - hist[0]));
		int th = 255;
		int sum = 0;
		for (int i = 1; i <= 255; i++)
		{
			sum += hist[i];
			if (sum > nr_non_muchie)
			{
				th = i;
				break;
			}
		}
		Mat bin_hist = Mat::zeros(src.size(), CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (mag_max.at<float>(i, j) < (K * th))
					bin_hist.at<uchar>(i, j) = 0;
				else if (mag_max.at<float>(i, j) >= (K * th) && mag_max.at<float>(i, j) <= th)
					bin_hist.at<uchar>(i, j) = 128;
				else
					bin_hist.at<uchar>(i, j) = 255;
		Mat ext_hist4 = bin_hist.clone();
		queue<pair<int, int>>Q;
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++)
				if (ext_hist4.at<uchar>(i, j) == 255)
					Q.push({ i,j });
		//parcurgerea pentru N4
		while (!Q.empty())
		{
			pair<int, int> p = Q.front();
			Q.pop();
			int dx[4] = { -1,0,1,0 };
			int dy[4] = { 0,1,0,-1 };
			for (int l = 0; l < 4; l++)
			{
				int inou, jnou;
				inou = p.first + dx[l];
				jnou = p.second + dy[l];
				if (isInside(src, inou, jnou))
				{
					if (ext_hist4.at<uchar>(inou, jnou) == 128)
					{
						ext_hist4.at<uchar>(inou, jnou) = 255;
						Q.push({ inou,jnou });
					}
				}
			}
		}
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (ext_hist4.at<uchar>(i, j) == 128)
					ext_hist4.at<uchar>(i, j) = 0;
		//parcurgerea pentru N8
		Mat ext_hist8 = bin_hist.clone();
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++)
				if (ext_hist8.at<uchar>(i, j) == 255)
					Q.push({ i,j });
		while (!Q.empty())
		{
			pair<int, int> p = Q.front();
			Q.pop();
			int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
			int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
			for (int l = 0; l < 8; l++)
			{
				int inou, jnou;
				inou = p.first + dx[l];
				jnou = p.second + dy[l];
				if (isInside(src, inou, jnou))
				{
					if (ext_hist8.at<uchar>(inou, jnou) == 128)
					{
						ext_hist8.at<uchar>(inou, jnou) = 255;
						Q.push({ inou,jnou });
					}
				}
			}
		}
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (ext_hist8.at<uchar>(i, j) == 128)
					ext_hist8.at<uchar>(i, j) = 0;
		imshow("Original", src);
		imshow("Gaussian filter", dstGauss);
		imshow("Magnitude", mag_norm);
		imshow("Magnitude without maximals", mag_max_norm);
		imshow("Adaptive treshold", bin_hist);
		imshow("Canny Result (N4)", ext_hist4);
		imshow("Canny Result (N8)", ext_hist8);
		waitKey();
	}
}
Mat drawContour(Mat src, int* label, Mat& c, Mat original)
{
	int height = src.rows;
	int width = src.cols;
	(*label) = 0;
	Mat labels = Mat::zeros(height, width, CV_32SC1);
	default_random_engine gen;
	uniform_int_distribution<int>d(0, 255);
	Vec3b color[1000];
	for (int i = 0; i < 1000; i++)
		color[i] = Vec3b(d(gen), d(gen), d(gen));
	int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
			{
				(*label)++;
				queue<pair<int, int>>Q;
				labels.at<int>(i, j) = (*label);
				Q.push({ i,j });
				while (!Q.empty())
				{
					pair<int, int> q = Q.front();
					Q.pop();
					int qi = q.first;
					int qj = q.second;
					for (int k = 0; k < 8; k++)
					{
						int inou = qi + dx[k];
						int jnou = qj + dy[k];
						if (isInside(src, inou, jnou))
						{
							if (src.at<uchar>(inou, jnou) == 0 && labels.at<int>(inou, jnou) == 0)
							{
								labels.at<int>(inou, jnou) = (*label);
								Q.push({ inou,jnou });
							}
						}
					}
				}
			}
		}
	}
	Mat aux, dst = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
	for (int l = 1; l <= (*label); l++)
	{
		int minX = width, maxX = 0, minY = height, maxY = 0;
		aux = Mat(height, width, CV_8UC1, 255);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) == l)
				{
					aux.at<uchar>(i, j) = 0;
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					minX = min(minX, j);
					maxX = max(maxX, j);
					minY = min(minY, i);
					maxY = max(maxY, i);
				}
			}
		}
		rectangle(original, Point(minX, minY), Point(maxX, maxY), Scalar(0, 0, 255), 1);
		//desenam conturul fiecarui obiect
		vector<pair<int, int>> contur;
		bool found = false;
		int dir;
		for (int i = 0; i < height && !found; i++)
		{
			for (int j = 0; j < width && !found; j++)
			{
				if (aux.at<uchar>(i, j) == 0)
				{
					found = true;
					//incepem conturul
					contur.push_back({ i,j });
					dst.at<Vec3b>(i, j) = color[l];
					c.at<uchar>(i, j) = 0;
					dir = 7;
					int inou, jnou;
					int x = i, y = j;
					while (true)
					{
						int startDir;
						if (dir % 2 == 0)
							startDir = (dir + 7) % 8;
						else
							startDir = (dir + 6) % 8;
						bool ok = false;
						for (int k = 0; k < 8; k++)
						{
							int currDir = (startDir + k) % 8;
							inou = x + dx[currDir];
							jnou = y + dy[currDir];
							if (isInside(aux, inou, jnou) && aux.at<uchar>(inou, jnou) == 0)
							{
								x = inou;
								y = jnou;
								contur.push_back({ inou,jnou });
								dst.at<Vec3b>(inou, jnou) = color[l];
								c.at<uchar>(inou, jnou) = 0;
								dir = currDir;
								ok = true;
								break;
							}
						}
						if (!ok)
							break;
						if (contur.size() >= 3 && contur[contur.size() - 1] == contur[1] && contur[contur.size() - 2] == contur[0])
						{
							break;
						}
					}
				}
			}
		}
	}
	return dst;
}
//creez o masca de fundal prin dilatare
void getBackground(const cv::Mat& source, cv::Mat& dst)
{
	cv::dilate(source, dst, cv::Mat::ones(3, 3, CV_8U));

}
//calculez transformata distanta pt a identifica centrul fiecarei celule
void getForeground(const cv::Mat& source, cv::Mat& dst)
{

	cv::distanceTransform(source, dst, cv::DIST_L2, 3, CV_32F);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
}
//gaseste contururi si creaza markeri pt watershed
void findMarker(const cv::Mat& sureBg, cv::Mat& markers, std::vector<std::vector<cv::Point>>& contours)
{
	cv::findContours(sureBg, contours, cv::RETR_EXTERNAL,
		cv::CHAIN_APPROX_SIMPLE);
	for (size_t i = 0, size = contours.size(); i < size; i++)
		drawContours(markers, contours, static_cast<int>(i),
			cv::Scalar(static_cast<int>(i) + 1), -1);
}

void getRandomColor(std::vector<cv::Vec3b>& colors, size_t size)
{
	for (int i = 0; i < size; ++i)
	{
		int b = cv::theRNG().uniform(0, 256);
		int g = cv::theRNG().uniform(0, 256);
		int r = cv::theRNG().uniform(0, 256);
		colors.emplace_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
}
void watershedSeparation(Mat& src) {
	Mat shifted = src.clone();
	//netezeste imaginea
	pyrMeanShiftFiltering(src, shifted, 15, 25);
	Mat gray_img;
	cvtColor(shifted, gray_img, COLOR_BGR2GRAY);
	//separarea automata a fundalului de obiecte
	Mat bin_img;
	threshold(gray_img, bin_img, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("Thresholded Image", bin_img);

	Mat sure_bg;
	dilate(bin_img, sure_bg, Mat::ones(3, 3, CV_8U));
	imshow("Sure Background", sure_bg);

	Mat dist;
	distanceTransform(bin_img, dist, DIST_L2, 3);
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	imshow("Distance Transform", dist);

	Mat dist_bin;
	threshold(dist, dist_bin, 0.35, 1.0, THRESH_BINARY);
	dist_bin.convertTo(dist_bin, CV_8U, 255);
	imshow("Sure Foreground", dist_bin);

	vector<vector<Point>> contours;
	Mat markers = Mat::zeros(dist_bin.size(), CV_32SC1);
	findContours(dist_bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	circle(markers, Point(5, 5), 3, Scalar(255), -1);

	Mat src_copy;
	if (src.channels() == 1)
		cvtColor(src, src_copy, COLOR_GRAY2BGR);
	else
		src.copyTo(src_copy);

	watershed(src_copy, markers);

	vector<Vec3b> colors(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);

	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int idx = markers.at<int>(i, j);
			if (idx > 0 && idx <= static_cast<int>(contours.size()))
				dst.at<Vec3b>(i, j) = colors[idx - 1];
			else if (idx == -1)
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
		}
	}

	imshow("Final Result", dst);
	printf("Numar de celule detectate: %zu\n", contours.size());
	Mat boxed = src.clone();
	set<int> labels;
	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			int val = markers.at<int>(i, j);
			if (val > 1) labels.insert(val);
		}
	}

	for (int label : labels) {
		Mat mask = markers == label;
		vector<vector<Point>> labelContours;
		findContours(mask, labelContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (const auto& cnt : labelContours) {
			if (cnt.size() > 10) {
				Rect box = boundingRect(cnt);
				rectangle(boxed, box, Scalar(0, 0, 255), 1);
			}
		}
	}
	imshow("Boxed Cells - Watershed implementation", boxed);
}
void project()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		//conversia din color in grayscale
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int blue, green, red;
				blue = src.at<Vec3b>(i, j)[0];
				green = src.at<Vec3b>(i, j)[1];
				red = src.at<Vec3b>(i, j)[2];
				int avg = (blue + green + red) / 3;
				gray.at<uchar>(i, j) = avg;
			}
		}
		imshow("Grayscale Image", gray);
		//aplicarea filtrului median(pentru eliminarea zgomotului de tip sare-piper
		Mat median = gray.clone();
		int w = 5;//window size
		int k = w / 2;//w=2*k+1 formula preluata din indrumatorul de laborator
		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{
				vector<uchar>v;
				for (int u = -k; u <= k; u++)
					for (int q = -k; q <= k; q++)
						v.push_back(gray.at<uchar>(i + u, j + q));
				sort(v.begin(), v.end());
				median.at<uchar>(i, j) = v[w * w / 2];//atribuim mediana
			}
		}
		imshow("Median filter applied", median);
		//aplicarea binarizarii adaptive
		float error = 0.3;
		int* h = returnHistogram(median);
		int imax = 255, imin = 0;
		while (h[imin] == 0 && imin < 255)
			imin++;
		while (h[imax] == 0 && imax > 0)
			imax--;
		double t1, t2;
		t1 = (imax + imin) / 2.0;
		t2 = t1;
		while (true) {
			t1 = t2;
			double g1, g2;
			g1 = g2 = 0;
			double n1 = 0, n2 = 0, sum1 = 0, sum2 = 0;
			for (int i = imin; i <= (int)t1; i++)
				n1 += h[i], sum1 += i * h[i];
			for (int i = (int)t1 + 1; i <= imax; i++)
				n2 += h[i], sum2 += i * h[i];
			if (n1 == 0 || n2 == 0) break;
			g1 = sum1 / n1;
			g2 = sum2 / n2;
			t2 = (g1 + g2) / 2.0;
			if (fabs(t2 - t1) < error)
				break;
		}
		Mat bin = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (median.at<uchar>(i, j) < t2)
					bin.at<uchar>(i, j) = 0;
				else
					bin.at<uchar>(i, j) = 255;
		cout << "Treshold: " << t2 << "\n";
		//imshow("Binarized image", bin);
		//etichetarea componentelor conexe, folosind bfs
		//pentru a vizualiza etichetarea, conturul fiecarei celule va avea o culoare diferita
		int label = 0;
		Mat c = Mat(height, width, CV_8UC1, 255);
		Mat boxed = src.clone();
		Mat contour = drawContour(bin, &label, c, boxed);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (contour.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
					contour.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		imshow("Colored contour for labeling visualization", contour);
		//imshow("Contour without objects", c);
		imshow("Boxed cells - Classic implementation", boxed);
		//printf("The number of cells found is: %d\n", label);
		watershedSeparation(src);
	}
}
int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf("13 - Additive factor\n");
		printf("14 - Multiplicative factor\n");
		printf("15 - 4 Colored Image\n");
		printf("16 - Inverse Matrix\n");
		printf("21 - Split image by RGB\n");
		printf("22 - Convert Color to Grayscale\n");
		printf("23 - Binarize Image\n");
		printf("24 - RGB to HSV\n");
		printf("25 - Verify is inside\n");
		printf("31 - Show histogram\n");
		printf("32 - Reduce Gray Levels + Floyd Steinberg\n");
		printf("41 - Compute Geometric Features\n");
		printf("51 - BFS N8\n");
		printf("52 - BFS N4\n");
		printf("53 - 2 Steps Crossing\n");
		printf("61 - Show N8 outline, chain code and derivative\n");
		printf("62 - Show Reconstructed Image\n");
		printf("71 - Show Morphology Operations Application\n");
		printf("72 - Contour and Fill\n");
		printf("81 - Show median, deviation, histogram, cummulative histogram\n");
		printf("82 - Binarized image based on error\n");
		printf("83 - Gamma Correction\n");
		printf("84 - Equalized Histogram\n");
		printf("91 - Apply filters\n");
		printf("92 - Logarithm, Fourier transform, Inverse Fourier transform,\n		Ideal Filters(low pass, high pass and gaussian)\n");
		printf("101 - Median filter applied for noise filtering\n");
		printf("102 - Gaussian 2D and 1D filter for noise filtering\n");
		printf("111 - Prewitt nucleus convolution, phase, magnitude and binarized magnitude\n");
		printf("112 - Gaussian filter and Sobel Nucleus\n");
		printf("150 - Project\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			additiveFactor();
			break;
		case 14:
			multiplicativeFactor();
			break;
		case 15:
			squaredImage();
			break;
		case 16:
			inverseMatrix();
			break;
		case 21:
			splitImageByRGB();
			break;
		case 22:
			colorToGrayScale();
			break;
		case 23:
			binarizeImage();
			break;
		case 24:
			convertRGBtoHSV();
			break;
		case 25:
			verifyIsInside();
			break;
		case 31:
			showAllHistograms();
			break;
		case 32:
			reduceGrayLevelsPlusFloydSteinberg();
			break;
		case 41:
			computeGeometricFeatures();
			break;
		case 51:
			BFSN8();
			break;
		case 52:
			BFSN4();
			break;
		case 53:
			twoStepsProcessing();
			break;
		case 61:
			showOutline();
			break;
		case 62:
			reconstructImage();
			break;
		case 71:
			morphologyOperations();
			break;
		case 72:
			showContourAndFillMorphologic();
			break;
		case 81:
			showHistogramProperties();
			break;
		case 82:
			binarizeImageBasedOnError();
			break;
		case 83:
			gammaCorrection();
			break;
		case 84:
			equalizeHistogram();
			break;
		case 91:
			applyFilters();
			break;
		case 92:
			generic_frequency_domain_filter();
			break;
		case 101:
			medianNoiseFiltering();
			break;
		case 102:
			gaussianNoiseFiltering();
			break;
		case 111:
			prewittNucleusApplication();
			break;
		case 112:
			gaussianFilterAndSobelNucleusApplication();
			break;
		case 150:
			project();
			break;
		}
	} while (op != 0);
	return 0;
}