/* openFileDlg() does not exist in linux */
/* int may need to be transformed into unsigned int */

#include <cstdio>
#include <cstdlib>
#include <queue>
#include <functional>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

const int patchSize = 15;        // unused in the demo, it's the Omega
const float omega = 0.95f;        // must be between 0 and 1 (used for maintaining a certain amount of haze, in order to make a natural image)
const float t0 = 0.1f;          // represents the lower bound of the transmission coefficient

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

uchar minPixel(Mat src, int y, int x)
{
    int offset = patchSize / 2;
    uchar res = 255;

    for(int i=y-offset; i<=y+offset; i++)
    {
        if(i < 0 || i >= src.rows) continue;
        for(int j=x-offset; j<=x+offset; j++)
        {
            if(j < 0 || j >= src.cols) continue;
            Vec3b pix = src.at<Vec3b>(i, j);
            uchar minPix = min(pix.val[0], min(pix.val[1], pix.val[2]));
            res = min(res, minPix);
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
            res.at<uchar>(i, j) = minPixel(src, i, j);
        }
    }

    return res;
}

uchar computeA(Mat dark)
{
    std::priority_queue<uchar, std::vector<uchar>, std::greater<uchar> > topVals;
    int total = 0;
    int number = dark.rows * dark.cols / 1000;

    for(int i=0; i<dark.rows; i++)
    {
        for(int j=0; j<dark.cols; j++)
        {
            uchar pix = dark.at<uchar>(i, j);
            if(topVals.size() < number)
            {
                topVals.push(pix);
                total += pix;
            }
            else
            {
                if(pix > topVals.top())
                {
                    total -= topVals.top();
                    topVals.pop();
                    topVals.push(pix);
                    total += pix;
                }
                else continue;
            }
        }
    }

    // topVals.clear();
    return (uchar)(total / number);
}

Mat computeRectified(Mat neg, Mat dark, uchar A)
{
    Mat res = Mat::zeros(neg.rows, neg.cols, CV_8UC3);
    for(int i=0; i<res.rows; i++)
    {
        for(int j=0; j<res.cols; j++)
        {
            float chosenT = max(1.0f - omega * dark.at<uchar>(i, j) / A, t0);
            Vec3b pix = neg.at<Vec3b>(i, j);
            Vec3b sol = pix;
            sol.val[0] = (uchar) min(((float)pix.val[0] - A) / chosenT + A, 255.0f);
            sol.val[1] = (uchar) min(((float)pix.val[1] - A) / chosenT + A, 255.0f);
            sol.val[2] = (uchar) min(((float)pix.val[2] - A) / chosenT + A, 255.0f);
            res.at<Vec3b>(i, j) = sol;
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
    Mat rectified;
    Mat enhanced;

    //
    uchar A;

    // while(openFileDlg(fname))
    // {
    //     src = imread(fname, CV_LOAD_IMAGE_COLOR);
    //     neg = negateImage(src);
    // }


    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    neg = negateImage(src);
    darkChannel = computeDarkChannel(neg);

    A = computeA(darkChannel);
    rectified = computeRectified(neg, darkChannel, A);
    // GaussianBlur(rectified, rectified, Size(5, 5), 7.0f);
    enhanced = negateImage(rectified);

    imshow("Source", src);
    // imshow("Negated", neg);
    // imshow("Dark channel", darkChannel);
    imshow("Rectified", rectified);
    imshow("Enhanced", enhanced);

    imwrite("output.jpg", enhanced);

    waitKey();

    return 0;
}
