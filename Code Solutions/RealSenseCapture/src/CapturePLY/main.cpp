#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <fstream>
#include <iomanip>

#include <iostream> 
using namespace std;
#include <chrono>
using namespace chrono;
#include <opencv2/opencv.hpp>
using namespace cv;

#include <thread>
#include <direct.h>

#define CAPTURES 50

int captureLoopImages() {
    _mkdir(".data/images");
    rs2::pipeline p;

    // Configure and start the pipeline
    p.start();

    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (auto i = 0; i < 30; ++i) p.wait_for_frames();

    rs2::frameset frames_ = p.wait_for_frames();
    rs2::depth_frame depth_ = frames_.get_depth_frame();
    auto width_ = depth_.get_width();
    auto height_ = depth_.get_height();

    float*** capturedImages = new float** [CAPTURES];

    auto start = high_resolution_clock::now();
    for (int i = 0; i < CAPTURES; i++)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get the depth frame's dimensions
        auto width = depth.get_width();
        auto height = depth.get_height();

        stringstream result;
        result << std::setprecision(0);

        float** image = new float* [height];
        for (int y = 0; y < height; y++) {
            image[y] = new float[width];
            for (int x = 0; x < width; x++) {
                image[y][x] = depth.get_distance(x, y);
            }
        }
        capturedImages[i] = image;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "100 imagenes capturadas en: " << duration.count() / 1000000.0 << "s" << endl;
    cout << "Guardando imagenes..." << endl;
    start = high_resolution_clock::now();

    for (int i = 0; i < CAPTURES; i++) {
        Mat imageToSave(height_, width_, CV_8UC3, Scalar(255, 255, 255));
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                Vec3b& color = imageToSave.at<Vec3b>(y, x);
                color[0] = (int)(capturedImages[i][y][x] * 33);
                color[1] = (int)(capturedImages[i][y][x] * 33);
                color[2] = (int)(capturedImages[i][y][x] * 33);
            }
        }
        stringstream filename;
        filename << ".data/images/Image_" << i << ".jpg";
        imwrite(filename.str(), imageToSave);
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "100 imagenes guardadas en: " << duration.count() / 1000000.0 << "s" << endl;

    p.stop();

    return 0;
}

int captureLoopPLY() {
    const int N = 100;
    const int timer = 500;

    system("pause");

    rs2::decimation_filter decimation_filter;
    rs2::pointcloud pointcloud;

    std::vector<rs2::points> points;
    std::vector<rs2::frame> colors, depths;

    rs2::pipeline pipe;
    pipe.start();

    std::cout << "Esperando 30 frames" << std::endl;
    for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();

    for (int i = 0; i < N; i++)
    {
        std::cout << i << " - Captura" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(timer));
        auto fs = pipe.wait_for_frames();

        auto depth = fs.get_depth_frame();
        auto color = fs.get_color_frame();

        depth.keep();
        color.keep();

        depths.push_back(depth);
        colors.push_back(color);

        pointcloud.map_to(colors[i]);
        depths[i] = decimation_filter.process(depths[i]);
        depths[i].keep();
        points.push_back(pointcloud.calculate(depths[i]));
        points[i].keep();
    }

    for (int i = 0; i < N; i++)
    {
        std::cout << i << " - Guardando frame " << depths[i].get_frame_number() << std::endl;
        stringstream filename;
        filename << ".data/ply/" << i << ".ply";
        points[i].export_to_ply(filename.str(), colors[i]);
    }

    return 0;
}

bool haveToPaintWhite(int x, int y, float height, float width, float centerDistance) {
    if (y == (int)(height / 2)) return true;
    if (x == (int)(width / 2)) return true;
    return false;
}

void captureImage(float top, float bottom, float right, float left, float inFront, float behind) {
}

int captureCenteredImage() {
    rs2::pipeline pipe;
    pipe.start();

    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();

    // Start your pipeline somewhere around here
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();
    auto color = frames.get_color_frame();
    auto height = depth.get_height();
    auto width = depth.get_width();

    // Query the distance from the camera to the object in the center of the image
    float dist_to_center = depth.get_distance(width / 2, height / 2);

    // Print the distance
    cout << "The camera is facing an object " << dist_to_center << " meters away \r";

    // Draw depth image with center marked
    Mat imageToSave(height, width, CV_8UC3, Scalar(255, 255, 255));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3b& color = imageToSave.at<Vec3b>(y, x);
            if (y == (int)(height / 2) || x == (int)(width / 2)) {
                color[0] = 255;
                color[1] = 255;
                color[2] = 255;
            }
            else {
                color[0] = (int)(depth.get_distance(x, y) * 33);
                color[1] = (int)(depth.get_distance(x, y) * 33);
                color[2] = (int)(depth.get_distance(x, y) * 33);
            }
        }
    }

    // Save image
    stringstream filename;
    filename << ".data/capturedImage.jpg";
    imwrite(filename.str(), imageToSave);

    rs2::pointcloud pc;
    pc.map_to(color);
    auto points = pc.calculate(depth);
    points.export_to_ply(".data/capturedImage.ply", color);

    return 0;
}

// Hello RealSense example demonstrates the basics of connecting to a RealSense device
// and taking advantage of depth data
int main(int argc, char* argv[]) try
{
    _mkdir(".data");

    return captureLoopPLY();
    return captureCenteredImage();
    return captureLoopImages();
    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
