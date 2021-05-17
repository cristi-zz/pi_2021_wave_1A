// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "common.h"
#include <stdio.h>
#include <vector>
#include <deque>


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image", src);
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

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i, j) = (r + g + b) / 3;
			}
		}

		imshow("original image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

bool inline isPowerOfTwo(const int number)
{
	return ((number & (number - 1)) == 0);
}

double computeAbsoluteMeanError(Mat_<double> original, Mat_<double> result)
{
	double mae = 0;
	for (auto i = 0; i < original.rows; i++)
	{
		for (auto j = 0; j < original.cols; j++)
		{
			mae += abs(original[i][j] - result[i][j]);
		}
	}
	mae /= (double)(original.rows * original.cols);
	return mae;
}

std::vector<double> getLowVector(const std::vector<double>& arr)
{
	std::vector<double> result(arr.size() / 2);
	for (auto i = 0; i < arr.size() / 2; i++)
	{
		result[i] = (arr[2 * i] + arr[2 * i + 1]) / 2.0f;
	}
	return result;
}

std::vector<double> getHighVector(const std::vector<double>& arr)
{
	std::vector<double> result(arr.size() / 2);
	for (auto i = 0; i < arr.size() / 2; i++)
	{
		result[i] = (arr[2 * i] - arr[2 * i + 1]) / 2.0f;
	}
	return result;
}

std::vector<double> getLowVectorUpSample(const std::vector<double>& arr)
{
	std::vector<double> result(2 * arr.size());;
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = arr[i / 2];
	}
	return result;
}

std::vector<double> getHighVectorUpSample(const std::vector<double>& arr)
{
	const int h[] = { 1, -1 };
	std::vector<double> result(2 * arr.size());;

	for (int i = 0; i < result.size(); i++)
	{
		result[i] = arr[i / 2] * h[i % 2];
	}
	return result;
}

std::vector<Mat_<double>> splitImage(const Mat_<double> src)
{
	Mat_<double> l = Mat_<double>(src.rows / 2, src.cols);
	Mat_<double> h = Mat_<double>(src.rows / 2, src.cols);

	Mat_<double> ll = Mat_<double>(src.rows / 2, src.cols / 2);
	Mat_<double> lh = Mat_<double>(src.rows / 2, src.cols / 2);
	Mat_<double> hl = Mat_<double>(src.rows / 2, src.cols / 2);
	Mat_<double> hh = Mat_<double>(src.rows / 2, src.cols / 2);

	for (auto j = 0; j < src.cols; j++)
	{
		std::vector<double> copy(src.rows);
		for (auto i = 0; i < src.rows; i++)
		{
			copy[i] = src[i][j];
		}

		auto low = getLowVector(copy);
		auto high = getHighVector(copy);

		for (auto i = 0; i < low.size(); i++)
		{
			l[i][j] = low[i];
			h[i][j] = high[i];
		}
	}

	for (auto i = 0; i < src.rows / 2; i++)
	{
		std::vector<double> copy_low(src.cols);
		std::vector<double> copy_high(src.cols);

		for (auto j = 0; j < src.cols; j++)
		{
			copy_low[j] = l[i][j];
			copy_high[j] = h[i][j];
		}

		auto low_low = getLowVector(copy_low);
		auto low_high = getHighVector(copy_low);

		auto high_low = getLowVector(copy_high);
		auto high_high = getHighVector(copy_high);

		for (auto j = 0; j < low_low.size(); j++)
		{
			ll[i][j] = low_low[j];
			lh[i][j] = low_high[j];

			hl[i][j] = high_low[j];
			hh[i][j] = high_high[j];
		}
	}

	return std::vector<Mat_<double>>{ll, lh, hl, hh};
}

Mat_<double> reconstructImage(const Mat_<double> ll, const Mat_<double> lh, const Mat_<double> hl, const Mat_<double> hh)
{
	const int32_t rows = ll.rows;
	const int32_t cols = ll.cols;

	Mat_<double> l(rows, 2 * cols);
	Mat_<double> h(rows, 2 * cols);

	Mat_<double> result(2 * rows, 2 * cols);

	for (auto i = 0; i < rows; i++)
	{
		std::vector<double> copy_low_low(cols);
		std::vector<double> copy_low_high(cols);

		std::vector<double> copy_high_low(cols);
		std::vector<double> copy_high_high(cols);

		for (auto j = 0; j < cols; j++)
		{
			copy_low_low[j] = ll[i][j];
			copy_low_high[j] = lh[i][j];

			copy_high_low[j] = hl[i][j];
			copy_high_high[j] = hh[i][j];
		}

		auto low_low_sample = getLowVectorUpSample(copy_low_low);
		auto low_high_sample = getHighVectorUpSample(copy_low_high);

		auto high_low_sample = getLowVectorUpSample(copy_high_low);
		auto high_high_sample = getHighVectorUpSample(copy_high_high);

		for (auto j = 0; j < low_low_sample.size(); j++)
		{
			l[i][j] = low_low_sample[j] + low_high_sample[j];
			h[i][j] = high_low_sample[j] + high_high_sample[j];
		}
	}

	for (int j = 0; j < 2 * cols; j++)
	{
		std::vector<double> copy_low(rows);
		std::vector<double> copy_high(rows);

		for (auto i = 0; i < rows; i++)
		{
			copy_low[i] = l[i][j];
			copy_high[i] = h[i][j];
		}

		auto low_sample = getLowVectorUpSample(copy_low);
		auto high_sample = getHighVectorUpSample(copy_high);

		for (auto i = 0; i < low_sample.size(); i++)
		{
			result[i][j] = low_sample[i] + high_sample[i];
		}
	}

	return result;
}

Mat_<double> reconstructImage(const std::vector<Mat_<double>> parts)
{
	return reconstructImage(parts[0], parts[1], parts[2], parts[3]);
}

std::deque<Mat_<double>> splitImageRecursive(const Mat_<double> src, const int min_rows = 16)
{
	std::deque<Mat_<double>> result;
	Mat_<double> ll = src.clone();

	while (ll.rows > min_rows)
	{
		auto parts = splitImage(ll);
		ll = parts[0].clone();

		for (auto i = parts.size() - 1; i >= 1; i--)
		{
			result.push_front(parts[i]);
		}
	}

	result.push_front(ll);
	return result;
}

Mat_<double> reconstructImageRecursive(std::deque<Mat_<double>> parts)
{
	auto ll = parts.front(); parts.pop_front();
	while (!parts.empty())
	{
		auto lh = parts.front(); parts.pop_front();
		auto hl = parts.front(); parts.pop_front();
		auto hh = parts.front(); parts.pop_front();

		ll = reconstructImage(ll, lh, hl, hh);
	}

	return ll;
}

void splitAndReconstruct()
{
	char fname[MAX_PATH] = {};
	while (openFileDlg(fname))
	{
		Mat_<double> src = imread(fname, IMREAD_GRAYSCALE);

		auto parts = splitImage(src);

		const Mat_<uint8_t> ll = parts[0];
		const Mat_<uint8_t> lh = parts[1];
		const Mat_<uint8_t> hl = parts[2];
		const Mat_<uint8_t> hh = parts[3];

		imshow("LL", ll);
		imshow("LH", lh + 128);
		imshow("HL", hl + 128);
		imshow("HH", hh + 128);

		Mat_<double> reconstructed = reconstructImage(parts);
		Mat_<uint8_t> reconstructed_int = reconstructed;
		imshow("Reconstructed", reconstructed_int);

		Mat_<uint8_t> original = src;
		imshow("Original", original);

		double mae = computeAbsoluteMeanError(src, reconstructed);
		printf("Mean absolute error: %f\n", mae);

		cv::waitKey(0);
	}
}

void displayParts(std::deque<Mat_<double>> parts)
{
	std::vector<std::pair<std::string, Mat_<uint8_t>>> img_array;
	char window_name[256] = {};
	int type = 0;

	for (auto i = 1; i < parts.size(); i++)
	{
		Mat_<uint8_t> to_show = parts[i] + 128;
		sprintf(window_name, "_%d_%d", to_show.rows, to_show.cols);

		switch (type)
		{
		case 0:
			img_array.push_back(std::make_pair("LH" + std::string(window_name), to_show));
			namedWindow("LH" + std::string(window_name), WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
			break;
		case 1:
			img_array.push_back(std::make_pair("HL" + std::string(window_name), to_show));
			namedWindow("HL" + std::string(window_name), WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
			break;
		case 2:
			img_array.push_back(std::make_pair("HH" + std::string(window_name), to_show));
			namedWindow("HH" + std::string(window_name), WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
			break;
		}
		type = (type + 1) % 3;
	}

	Mat_<uint8_t> ll = parts[0];
	namedWindow("LL", WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
	imshow("LL", ll);



	for (auto& it : img_array)
	{
		imshow(it.first, it.second);
	}
}

void splitAndReconstructRecursive()
{
	char fname[MAX_PATH] = {};
	int min_rows = 0;

	while (openFileDlg(fname))
	{
		Mat_<double> src = imread(fname, IMREAD_GRAYSCALE);
		printf("Minimum rows: "); scanf("%d", &min_rows);
		auto parts = splitImageRecursive(src, min_rows);

		Mat_<double> reconstructed = reconstructImageRecursive(parts);
		Mat_<uint8_t> reconstructed_int = reconstructed;
		imshow("Reconstructed", reconstructed_int);

		Mat_<uint8_t> original = src;
		imshow("Original", original);

		displayParts(parts);

		double mae = computeAbsoluteMeanError(src, reconstructed);
		printf("Mean absolute error: %f\n", mae);

		cv::waitKey(0);
	}
}

void filterH(Mat_<double> src, int threshold)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src[i][j] < threshold) {
				src[i][j] = 0;
			}
		}
	}
}

void splitAndReconstructFilter() {
	char fname[MAX_PATH] = {};
	int threshold = 0;
	int size = 0;

	while (openFileDlg(fname))
	{
		printf("Minimum rows: "); scanf("%d", &size);
		printf("Threshold: "); scanf("%d", &threshold);
		Mat_<double> src = imread(fname, IMREAD_GRAYSCALE);
		auto parts = splitImageRecursive(src, size);

		for (int i = 1; i < parts.size(); i++)
		{
			filterH(parts[i], threshold);
		}

		displayParts(parts);

		Mat_<double> reconstructed = reconstructImageRecursive(parts);
		Mat_<uint8_t> reconstructed_int = reconstructed;
		imshow("Reconstructed", reconstructed_int);

		Mat_<uint8_t> original = src;
		imshow("Original", original);

		Mat_<uint8_t> dst(original.rows, original.cols);
		double avg = 0.0f;
		for (int i = 0; i < original.rows; i++)
		{
			for (int j = 0; j < original.cols; j++)
			{
				dst[i][j] = abs(original[i][j] - reconstructed_int[i][j]);
				avg += dst[i][j];
			}
		}
		avg /= (dst.rows * dst.cols);

		imshow("Original-Reconstructed", dst);
		printf("Average: %.2f\n", avg);

		cv::waitKey(0);
	}
}

void swapLLRecursive() {
	char fname1[MAX_PATH] = {};
	char fname2[MAX_PATH] = {};
	int min_rows = 0;

	if (openFileDlg(fname1) && openFileDlg(fname2))
	{
		Mat_<double> src1 = imread(fname1, IMREAD_GRAYSCALE);
		Mat_<double> src2 = imread(fname2, IMREAD_GRAYSCALE);

		printf("Src1: %dx%d\n", src1.rows, src1.cols);
		printf("Src2: %dx%d\n", src2.rows, src2.cols);

		printf("Minimum rows: "); scanf("%d", &min_rows);

		auto parts1 = splitImageRecursive(src1, min_rows);
		auto parts2 = splitImageRecursive(src2, min_rows);

		// swap LLs
		auto aux = parts1[0];
		parts1[0] = parts2[0];
		parts2[0] = aux;


		// reconstruire
		Mat_<double> reconstructed1 = reconstructImageRecursive(parts1);
		Mat_<uint8_t> reconstructed_int1 = reconstructed1;
		imshow("Reconstructed imag 1", reconstructed_int1);

		Mat_<double> reconstructed2 = reconstructImageRecursive(parts2);
		Mat_<uint8_t> reconstructed_int2 = reconstructed2;
		imshow("Reconstructed imag 2", reconstructed_int2);

		Mat_<uint8_t> original1 = src1;
		imshow("Original imag 1", original1);

		Mat_<uint8_t> original2 = src2;
		imshow("Original imag 2", original2);

		cv::waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("1 - Split and reconstruct an image\n");
		printf("2 - Split and reconstruct an image (recursively)\n");
		printf("3 - Split and reconstruct an image + filter *H* (recursively)\n");
		printf("4 - Swap LLs and reconstruct (recursively)\n");
		scanf("%d", &op);

		switch (op)
		{
		case 1:
			splitAndReconstruct();
			break;

		case 2:
			splitAndReconstructRecursive();
			break;

		case 3:
			splitAndReconstructFilter();
			break;

		case 4:
			swapLLRecursive();
			break;
		}
	} while (op != 0);

	return 0;
}
