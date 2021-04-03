// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "common.h"
#include <stdio.h>
#include <vector>

//#define ASSERTS_ON


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

#ifdef ASSERTS_ON
	assert(parts.size() == 4);
	for (auto& i : parts)
	{
		for (auto& j : parts)
		{
			assert(i.cols == j.cols);
			assert(i.rows == j.rows);
		}
	}
#endif // ASSERTS_ON

	return reconstructImage(parts[0], parts[1], parts[2], parts[3]);
}

void splitAndReconstruct()
{
	char fname[MAX_PATH] = {};
	while (openFileDlg(fname))
	{
		Mat_<double> src = imread(fname, IMREAD_GRAYSCALE);

#ifdef ASSERTS_ON
		assert(isPowerOfTwo(src.rows));
		assert(isPowerOfTwo(src.cols));
#endif // ASSERTS_ON

		auto parts = splitImage(src);

		const Mat_<uint8_t> ll = parts[0];
		const Mat_<uint8_t> lh = parts[1];
		const Mat_<uint8_t> hl = parts[2];
		const Mat_<uint8_t> hh = parts[3];

		imshow("LL", ll);
		imshow("LH", lh);
		imshow("HL", hl);
		imshow("HH", hh);

		Mat_<uint8_t> reconstructed = reconstructImage(parts);
		imshow("Reconstructed", reconstructed);

		Mat_<uint8_t> original = src;
		imshow("Original", original);

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
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			splitAndReconstruct();
			break;
		}
	} 	while (op != 0);
	return 0;
}
