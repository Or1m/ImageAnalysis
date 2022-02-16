#include "stdafx.h"
#include <random>
#include <iostream>

void floodFill(cv::Mat& img, const cv::Vec3b& newColor, int y, int x);
void floodFillUtil(cv::Mat& img, int x, int y, cv::Vec3b& prevColor, const cv::Vec3b& newColor);

int main()
{
	const cv::Vec3b whitePoint = cv::Vec3b(255, 255, 255);
	const cv::Vec3b blackPoint = cv::Vec3b(0, 0, 0);

	const cv::String filename = "images/train.png";
	const uchar treshold = 1;
	
	cv::Vec3b newColor;
	cv::Mat img, indexedImg;

	std::random_device rd; // obtain a random number from hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> distr(0, 255); // define the range


	// load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, indexedImg, cv::COLOR_BGR2GRAY);

	if (img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return EXIT_FAILURE;
	}

	for (int y = 0; y < img.rows; y++) {

		for (int x = 0; x < img.cols; x++) {

			if (img.at<cv::Vec3b>(y, x)[0] > treshold || 
				img.at<cv::Vec3b>(y, x)[1] > treshold || 
				img.at<cv::Vec3b>(y, x)[2] > treshold) {

				img.at<cv::Vec3b>(y, x) = whitePoint;
			}
			else {
				img.at<cv::Vec3b>(y, x) = blackPoint;
			}
		}
	}

	/*for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			std::cout << (int)img.at<uchar>(y, x) << " ";
		}
	}*/

	std::vector<cv::Vec3b> colors;
	
	for (int y = 0; y < img.rows; y++) {

		for (int x = 0; x < img.cols; x++) {

			if (img.at<cv::Vec3b>(y, x) == whitePoint) {

				newColor = cv::Vec3b(distr(gen), distr(gen), distr(gen));
				colors.emplace_back(newColor);

				floodFill(img, newColor, y, x);
			}
		}
	}

	for (int y = 0; y < img.rows; y++) {

		for (int x = 0; x < img.cols; x++) {

			if (img.at<cv::Vec3b>(y, x) == whitePoint) {

				newColor = cv::Vec3b(distr(gen), distr(gen), distr(gen));
				colors.emplace_back(newColor);

				floodFill(img, newColor, y, x);
			}
		}
	}

	// diplay image
	cv::imshow("Separated Image", img);
	// wait until keypressed
	cv::waitKey(0); 

	return EXIT_SUCCESS;
}

void floodFill(cv::Mat& img, const cv::Vec3b& newColor, int y, int x)
{
	cv::Vec3b prevColor = img.at<cv::Vec3b>(y, x);

	if (prevColor == newColor)
		return;

	floodFillUtil(img, x, y, prevColor, newColor);
}

void floodFillUtil(cv::Mat& img, int x, int y, cv::Vec3b& prevColor, const cv::Vec3b& newColor)
{
	// Base cases
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (img.at<cv::Vec3b>(y, x) != prevColor)
		return;
	if (img.at<cv::Vec3b>(y, x) == newColor)
		return;

	// Replace the color at (x, y)
	img.at<cv::Vec3b>(y, x) = newColor;

	// Recur for north, east, south and west
	floodFillUtil(img, x + 1, y, prevColor, newColor);
	floodFillUtil(img, x - 1, y, prevColor, newColor);
	floodFillUtil(img, x, y + 1, prevColor, newColor);
	floodFillUtil(img, x, y - 1, prevColor, newColor);
}
