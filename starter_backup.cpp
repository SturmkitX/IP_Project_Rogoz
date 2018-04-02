/* openFileDlg() does not exist in linux */

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

const int patchSize = 1;        // unused in the demo, it's the Omega
const int minThreshold = 20;    // T

Mat negateImage(Mat src)
{
    Mat res = Mat::zeros(src.rows, src.cols, CV_8UC3);
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            Vec3b pix = src.at<Vec3b>(i, j);
            pix.val[0] = 255 - pix.val[0];
            pix.val[1] = 255 - pix.val[1];
            pix.val[2] = 255 - pix.val[2];
            res.at<Vec3b>(i, j) = pix;
        }
    }

    return res;
}

Mat computeDarkChannel(Mat src)
{
    Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            Vec3b pix = src.at<Vec3b>(i, j);
            res.at<uchar>(i, j) = min(min(pix.val[0], pix.val[1]), pix.val[2]);
        }
    }

    return res;
}

uint *computeHistogram(Mat src)
{
    uint *res = (uint*) calloc(256, sizeof(uint));
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            uchar pix = src.at<uchar>(i, j);
            res[pix]++;
        }
    }

    return res;
}

int main(int argc, char **argv)
{
    // char fname[100];
    Mat src;
    Mat neg;
    Mat darkChannel;

    //
    uint *histogram;
    Vec3b A;

    // while(openFileDlg(fname))
    // {
    //     src = imread(fname, CV_LOAD_IMAGE_COLOR);
    //     neg = negateImage(src);
    // }
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    neg = negateImage(src);
    darkChannel = computeDarkChannel(neg);
    histogram = computeHistogram(darkChannel);
    A = computeA(histogram, darkChannel);

    imshow("Original", src);
    imshow("Negated", neg);

    waitKey();
    free(histogram);
    return 0;
}
