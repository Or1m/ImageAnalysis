#include "stdafx.h"
#include "Header.h"

int main(void) {
	
	// Declarations
	const cv::String trainFilename	= "images/train.png";
	const cv::String testFilename	= "images/test01.png";
	
	const EClassificationType classificationType = EClassificationType::BNN;
	const uchar threshold = 220;

	cv::Mat sourceTrainImg, grayScaleTrainImg;
	cv::Mat sourceTestImg, grayScaleTestImg;
	
	// Loading & converting
	sourceTrainImg = cv::imread(trainFilename, CV_LOAD_IMAGE_COLOR);
	sourceTestImg = cv::imread(testFilename, CV_LOAD_IMAGE_COLOR);

	cv::cvtColor(sourceTrainImg, grayScaleTrainImg, cv::COLOR_BGR2GRAY);
	cv::cvtColor(sourceTestImg, grayScaleTestImg, cv::COLOR_BGR2GRAY);

	if (sourceTrainImg.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return EXIT_FAILURE;
	}

	// Processing of the train image
	ApplyThresholding(&sourceTrainImg, grayScaleTrainImg, threshold);

	int objectCount = ApplyFloodFill(&sourceTrainImg, grayScaleTrainImg);

	std::vector<FeatureVector> features;
	CalcFeatures(objectCount, sourceTrainImg, grayScaleTrainImg, features);

	// Processing of the test image
	ApplyThresholding(nullptr, grayScaleTestImg, threshold, true);
	int testObjectCount = ApplyFloodFill(nullptr, grayScaleTestImg, true);

	std::vector<FeatureVector> testFeatures;
	CalcFeatures(testObjectCount, sourceTestImg, grayScaleTestImg, testFeatures);
	
	// Classification
	if (classificationType == EClassificationType::Ethalons) {
		std::map<EClassType, FeatureVector> ethalons;
		CalcEthalons(features, ethalons);
		CompareFeaturesWithEthalons(ethalons, testFeatures, sourceTestImg);
	}
	else if (classificationType == EClassificationType::KMeans) {
		std::map<FeatureVector, std::vector<FeatureVector>> clusters;
		DoClusteringUntilCorrectResult(clusters, features);
		CompareFeaturesWithCentroids(clusters, testFeatures, sourceTestImg);
	}
	else {
		NN* nn = createNN(2, 4, 3);
		
		train(nn, features);
		test(nn, testFeatures, sourceTestImg);
		
		releaseNN(nn);
	}

	// Debug output
	/*for (int y = 0; y < grayScaleImg.rows; y++) {
		for (int x = 0; x < grayScaleImg.cols; x++) {
			std::cout << (int)grayScaleImg.at<uchar>(y, x) << " ";
		}
	}*/

	// Display images
	cv::imshow("Colored Image", sourceTrainImg);
	cv::imshow("Test Image", sourceTestImg);

	cv::waitKey(0); 
	return EXIT_SUCCESS;
}

// Thresholding
void ApplyThresholding(cv::Mat* coloredImg, cv::Mat& grayScaleImg, const uchar threshold, bool grayscaleOnly) {
	
	for (int y = 0; y < grayScaleImg.rows; y++) {

		for (int x = 0; x < grayScaleImg.cols; x++) {

			if (grayScaleImg.at<uchar>(y, x) > threshold) {

				if (!grayscaleOnly)
					(*coloredImg).at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);

				grayScaleImg.at<uchar>(y, x) = 255;
			}
			else {

				if (!grayscaleOnly)
					(*coloredImg).at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);

				grayScaleImg.at<uchar>(y, x) = 0;
			}
		}
	}
}

// FloodFill
int ApplyFloodFill(cv::Mat* coloredImg, cv::Mat& grayScaleImg, bool grayScaleOnly) {
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> distr(0, 255); // define the range

	std::vector<cv::Vec3b> colors;
	cv::Vec3b newColor;
	uchar index = 1;

	for (int y = 0; y < grayScaleImg.rows; y++) {

		for (int x = 0; x < grayScaleImg.cols; x++) {

			//if (sourceImg.at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255)) {
			if (grayScaleImg.at<uchar>(y, x) == 255) {

				if (!grayScaleOnly) {
					newColor = cv::Vec3b(distr(gen), distr(gen), distr(gen));
					colors.emplace_back(newColor);

					floodFill(*coloredImg, newColor, y, x);
				}

				floodFillPrimitive(grayScaleImg, index++, y, x);
			}
		}
	}

	std::cout << "MAX COLORS: " << colors.size() << std::endl;
	std::cout << "MAX INDEX: " << (int)index - 1 << std::endl;

	return index - 1;
}

template <typename T>
void floodFill(cv::Mat& img, T& newColor, int y, int x) {
	T prevColor = img.at<T>(y, x);

	if (prevColor == newColor)
		return;

	floodFillUtil(img, x, y, prevColor, newColor);
}
template <typename T>
void floodFillUtil(cv::Mat& img, int x, int y, T& prevColor, T& newColor) {
	// Base cases
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (img.at<T>(y, x) != prevColor)
		return;
	if (img.at<T>(y, x) == newColor)
		return;

	// Replace the color at (x, y)
	img.at<T>(y, x) = newColor;

	// Recur for north, east, south and west
	floodFillUtil(img, x + 1, y, prevColor, newColor);
	floodFillUtil(img, x - 1, y, prevColor, newColor);
	floodFillUtil(img, x, y + 1, prevColor, newColor);
	floodFillUtil(img, x, y - 1, prevColor, newColor);

	floodFillUtil(img, x + 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x + 1, y - 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y - 1, prevColor, newColor);
}

void floodFillPrimitive(cv::Mat& img, uchar newColor, int y, int x) {
	uchar prevColor = img.at<uchar>(y, x);

	if (prevColor == newColor)
		return;

	floodFillUtilPrimitive(img, x, y, prevColor, newColor);
}

void floodFillUtilPrimitive(cv::Mat& img, int x, int y, uchar prevColor, uchar newColor) {
	// Base cases
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (img.at<uchar>(y, x) != prevColor)
		return;
	if (img.at<uchar>(y, x) == newColor)
		return;

	// Replace the color at (x, y)
	img.at<uchar>(y, x) = newColor;

	// Recur for north, east, south and west
	floodFillUtil(img, x + 1, y, prevColor, newColor);
	floodFillUtil(img, x - 1, y, prevColor, newColor);
	floodFillUtil(img, x, y + 1, prevColor, newColor);
	floodFillUtil(img, x, y - 1, prevColor, newColor);

	floodFillUtil(img, x + 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x + 1, y - 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y - 1, prevColor, newColor);
}

// Features
void CalcFeatures(int objectCount, cv::Mat& sourceImg, cv::Mat& grayScaleImg, std::vector<FeatureVector>& features) {
	
	for (int i = 1; i <= objectCount; i++) {

		double m10 = 0, m01 = 0, m00 = 0;

		for (int y = 0; y < grayScaleImg.rows; y++) {

			for (int x = 0; x < grayScaleImg.cols; x++) {

				if (grayScaleImg.at<uchar>(y, x) == i) {

					m10 += M(1, 0, x, y);
					m01 += M(0, 1, x, y);
					m00 += M(0, 0, x, y);
				}
			}
		}

		double xt = m10 / (double)m00;
		double yt = m01 / (double)m00;

		double area = 0; // == m00
		double circumference = 0;

		double mu20 = 0, mu02 = 0, mu11 = 0;

		for (int y = 0; y < grayScaleImg.rows; y++) {

			for (int x = 0; x < grayScaleImg.cols; x++) {

				if (grayScaleImg.at<uchar>(y, x) == i) {

					area += Mu(0, 0, x, xt, y, yt);
					mu20 += Mu(2, 0, x, xt, y, yt);
					mu02 += Mu(0, 2, x, xt, y, yt);
					mu11 += Mu(1, 1, x, xt, y, yt);

					if (IsPartOfCircumference(grayScaleImg, y, x, i))
						circumference += 1;
				}
			}
		}

		double f1 = pow(circumference, 2) / (100 * area);
		double f2 = MuMin(mu20, mu02, mu11) / MuMax(mu20, mu02, mu11);

		std::cout << "Object " << i
			<< ": Area: " << area << ", Circumference: " << circumference
			<< ", F1: " << f1 << ", F2: " << f2 << std::endl; // F2 bad

		features.push_back({ f1, f2, xt, yt });
	}

	std::cout << std::endl;
}

void AddLabelToImage(cv::Mat& grayScaleImg, int xt, int yt, int i, int colorClass) {

	auto color = CV_RGB(0, 0, 0);
	int offset = textHeight * 10;

	switch (colorClass) {
		case 1:
			color = CV_RGB(255, 0, 128);
			break;
		case 2:
			color = CV_RGB(165, 255, 0);
			break;
		case 3:
			color = CV_RGB(0, 255, 255);
			break;
	}

	cv::putText(grayScaleImg, std::to_string(i), cv::Point(xt - offset, yt + offset), cv::FONT_HERSHEY_DUPLEX, textHeight, color, 2);
}

inline bool IsPartOfCircumference(cv::Mat& grayScaleImg, int y, int x, int i) {
	
	return
		grayScaleImg.at<uchar>(y + 1, x) != i ||
		grayScaleImg.at<uchar>(y - 1, x) != i ||
		grayScaleImg.at<uchar>(y, x + 1) != i ||
		grayScaleImg.at<uchar>(y, x - 1) != i;
}

inline double Mu(int p, int q, int x, double xt, int y, double yt) {
	return pow(x - xt, p) * pow(y - yt, q); // * f(x,y) == 1
}

inline double MuMax(double mu20, double mu02, double mu11) {
	return (mu20 + mu02) / 2 + sqrt(4 * pow(mu11, 2) + pow(mu20 - mu02, 2)) / 2;
}

inline double MuMin(double mu20, double mu02, double mu11) {
	return (mu20 + mu02) / 2 - sqrt(4 * pow(mu11, 2) + pow(mu20 - mu02, 2)) / 2;
}

inline double M(int p, int q, int x, int y) {
	return pow(x, p) * pow(y, q); // * f(x,y) == 1
}

// Labeling, ethalons, comparing to test image
void CalcEthalons(std::vector<FeatureVector>& features, std::map<EClassType, FeatureVector>& ethalons) {
	
	EClassType tempType = EClassType::Square;

	for (int i = 0; i < features.size(); i++)
	{
		const auto& featureVector = features[i];

		if (i >= starIdx)
			tempType = EClassType::Star;
		if (i >= rectangleIdx)
			tempType = EClassType::Rectangle;

		ethalons[tempType].f1 += featureVector.f1;
		ethalons[tempType].f2 += featureVector.f2;
	}

	for (auto& [key, value] : ethalons) {

		if (key == EClassType::Square) {
			value.f1 /= numOfSquares;
			value.f2 /= numOfSquares;
		}
		else if (key == EClassType::Star)
		{
			value.f1 /= numOfStars;
			value.f2 /= numOfStars;
		}
		else if (key == EClassType::Rectangle)
		{
			value.f1 /= numOfRectangles;
			value.f2 /= numOfRectangles;
		}
	}
}

void CompareFeaturesWithEthalons(const std::map<EClassType, FeatureVector>& ethalons, const std::vector<FeatureVector>& testFeatures, cv::Mat& sourceImg) {
	
	std::cout << "Testing image:" << std::endl;
	
	int iterator = 1;

	for (auto& feature : testFeatures) {

		double minDistance = std::numeric_limits<double>::max();
		EClassType currentType = EClassType::Unknown;

		for (auto& [key, value] : ethalons) {

			double distance = Euklid(feature, value);

			if (distance < minDistance) {
				minDistance = distance;
				currentType = key;
			}
		}

		AddLabelToImage(sourceImg, (int)feature.xt, (int)feature.yt, iterator, (int)currentType);
		std::cout << "Object " << iterator++ << ": " << currentType << std::endl;
	}
}

// KMeans
void DoClusteringUntilCorrectResult(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, std::vector<FeatureVector>& features) {
	
	bool correctFlag;

	do {
		correctFlag = true;
		ComputeKMeansClustering(clusters, features);

		for (auto& [key, value] : clusters) {

			if (value.size() != expectedCount) {
				correctFlag = false;
				break;
			}
		}

	} while (!correctFlag);
}

void ComputeKMeansClustering(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, const std::vector<FeatureVector>& trainFeatures) {
	
	std::vector<FeatureVector> centroids;
	std::vector<FeatureVector> tempCentroids;

	ChooseInitialCentroids(centroids, trainFeatures, (int)trainFeatures.size());

	AssignFeaturesToNearestCentroids(trainFeatures, centroids, clusters);

	while (true) {
		
		RecalculateCentroids(tempCentroids, centroids, clusters);

		int a = 3;
		if (VectorsEqual(tempCentroids, centroids))
			break;
		else
			centroids = tempCentroids; //Nepreda sa len referencia?
			
		AssignFeaturesToNearestCentroids(trainFeatures, centroids, clusters);
	}
}

void ChooseInitialCentroids(std::vector<FeatureVector>& centroids, const std::vector<FeatureVector>& trainFeatures, int length) {
	
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> distr(0, length); // define the range

	std::vector<int> possibles;
	for (int i = 0; i < length; i++)
		possibles.push_back(i);

	std::shuffle(possibles.begin(), possibles.end(), gen);

	for (int i = 0; i < k; i++)
		centroids.push_back(trainFeatures[possibles[i]]);
}

void AssignFeaturesToNearestCentroids(const std::vector<FeatureVector>& trainFeatures, std::vector<FeatureVector>& centroids, std::map<FeatureVector, std::vector<FeatureVector>>& clusters) {
	
	clusters.clear();

	for (auto& feature : trainFeatures) {
		FeatureVector nearestCentroid;
		double nearestCentroidDistance = std::numeric_limits<double>::max();

		for (auto& centroid : centroids) {
			double distance = Euklid(feature, centroid);

			if (distance < nearestCentroidDistance) {
				nearestCentroidDistance = distance;
				nearestCentroid = centroid;
			}
		}

		clusters[nearestCentroid].push_back(feature);
	}
}

void RecalculateCentroids(std::vector<FeatureVector>& tempCentroids, std::vector<FeatureVector>& centroids, std::map<FeatureVector, std::vector<FeatureVector>>& clusters) {
	
	tempCentroids.clear();

	for (auto& centroid : centroids) {
		FeatureVector avgs(0, 0);

		for (auto& assignedFeatures : clusters[centroid]) {
			avgs.f1 += assignedFeatures.f1;
			avgs.f2 += assignedFeatures.f2;
		}

		avgs.f1 /= clusters[centroid].size();
		avgs.f2 /= clusters[centroid].size();

		tempCentroids.push_back(avgs);
	}
}

void CompareFeaturesWithCentroids(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, std::vector<FeatureVector>& testFeatures, cv::Mat& sourceImg) {
	
	std::cout << "Testing image:" << std::endl;
	std::map<int, FeatureVector> classes;

	int iterator = 1;
	for (auto& [key, value] : clusters)
		classes[iterator++] = key;

	iterator = 1;
	for (auto& feature : testFeatures) {

		double minDistance = std::numeric_limits<double>::max();
		int objClass = -1;

		for (auto& [key, value] : classes) {

			double distance = Euklid(feature, value);

			if (distance < minDistance) {
				minDistance = distance;
				objClass = key;
			}
		}

		AddLabelToImage(sourceImg, (int)feature.xt, (int)feature.yt, iterator, objClass);
		std::cout << "Object " << iterator++ << " belongs to class: " << objClass << std::endl;
	}
}

//BNN
void train(NN* nn, std::vector<FeatureVector>& features) {
	int n = features.size();
	double** trainingSet = new double* [n];
	double y1 = 1.0, y2 = 0.0, y3 = 0.0;

	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		if (i >= starIdx) {
			y1 = 0.0;
			y2 = 0.0;
			y3 = 1.0;
		}
		if (i >= rectangleIdx) {
			y1 = 0.0;
			y2 = 1.0;
			y3 = 0.0;
		}

		int key = 0;
		for (auto value : { features[i].f1, features[i].f2, y1, y2, y3 }) {
			trainingSet[i][key] = value;
			++key;
		}
	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i % n], true);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, std::vector<FeatureVector>& features, cv::Mat& sourceImg) {
	int n = features.size();
	double* in = new double[nn->n[0]];

	for (int i = 0; i < n; i++) {
		int key = 0;
		FeatureVector feature = features[i];

		for (auto value : { feature.f1, feature.f2 }) {
			in[key] = value;
			key++;
		}

		setInput(nn, in, true);
		feedforward(nn);

		bool verbose = true;
		int output = getOutput(nn, verbose, i + 1);

		if (verbose)
			std::cout << " -> " << (EClassType)(output + 1) << std::endl;

		AddLabelToImage(sourceImg, (int)feature.xt, (int)feature.yt, i + 1, output + 1);
	}
}

// Utils
double Euklid(const FeatureVector& feature, const FeatureVector& ethalon) {
	return sqrt(pow(feature.f1 - ethalon.f1, 2) + pow(feature.f2 - ethalon.f2, 2));
}

std::ostream& operator<<(std::ostream& os, const EClassType type) {
	
	switch (type) {

		case EClassType::Square:
			os << "Square";
			break;
		case EClassType::Star:
			os << "Star";
			break;
		case EClassType::Rectangle:
			os << "Rectangle";
			break;
	}

	return os;
}

template<typename T>
bool VectorsEqual(std::vector<T>& v1, std::vector<T>& v2) {
	
	std::sort(v1.begin(), v1.end());
	std::sort(v2.begin(), v2.end());
	
	return v1 == v2;
}
