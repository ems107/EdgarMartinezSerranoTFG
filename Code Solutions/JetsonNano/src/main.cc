#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define _WINDOWS_
#else
#define _LINUX_
#endif

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm> 

// Jsoncpp
#include <json/json.h>

// RealSense
#include <librealsense2/rs.hpp>

// Point Cloud Library
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/statistical_outlier_removal.h>

#ifdef _WINDOWS_
#include <direct.h>
#endif

#ifdef _LINUX_
#include <sys/stat.h>

// CUDA-ICP
#include "lib/cudaICP.h"
#endif

// Json Parameters
std::string DataPath = "";
int RealSenseCaptureTotalTime_ms = 0;
int RealSenseCaptureTimingBetweenCaptures_ms = 0;
int NumberOfCaptureFiles = 0;
float ResizeCenterDistance_m = 0;
float ResizeCaptureBottom_m = 0;
float ResizeCaptureTop_m = 0;
float ResizeCaptureLeft_m = 0;
float ResizeCaptureRight_m = 0;
float ResizeCaptureBehind_m = 0;
float ResizeCaptureFront_m = 0;
int ICP_Iterations = 0;
float ICP_MaxCorrespondenceDistance = 0;

void createDirectories();
void customMkdir(std::string);
void readParametersFromJson();
float getDistanceToCenter();
void print4x4Matrix(const Eigen::Matrix4f&);
void adquisitionFormRealSense(bool);
void adquisitionFromRealSenseLoop(bool);
void applyPreprocessingFilterTo(pcl::PointCloud<pcl::PointXYZRGB>::Ptr&);
void applyPreprocessingFilterToRealSenseCaptures();
void pointCloudReconstructionUsingICP(bool, bool);
Eigen::Matrix4f applyCudaICP(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&, pcl::PointCloud<pcl::PointXYZRGB>::Ptr&);
Eigen::Matrix4f applyICP(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&, pcl::PointCloud<pcl::PointXYZRGB>::Ptr&);
pcl::PointCloud<pcl::PointXYZ>::Ptr parseToPCL(const rs2::points&);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr parseToPCLRGB(const rs2::points&, const rs2::video_frame&);
std::tuple<int, int, int> getTextureRGB(rs2::video_frame, rs2::texture_coordinate);

std::tuple<int, int, int> getTextureRGB(rs2::video_frame texture, rs2::texture_coordinate Texture_XY)
{
    // Get Width and Height coordinates of texture
    int width = texture.get_width();  // Frame width in pixels
    int height = texture.get_height(); // Frame height in pixels

    // Normals to Texture Coordinates conversion
    int x_value = std::min(std::max(int(Texture_XY.u * width + .5f), 0), width - 1);
    int y_value = std::min(std::max(int(Texture_XY.v * height + .5f), 0), height - 1);

    int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
    int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
    int Text_Index = (bytes + strides);

    const auto New_Texture = reinterpret_cast<const uint8_t*>(texture.get_data());

    // RGB components to save in tuple
    int NT1 = New_Texture[Text_Index];
    int NT2 = New_Texture[Text_Index + 1];
    int NT3 = New_Texture[Text_Index + 2];

    return std::tuple<int, int, int>(NT1, NT2, NT3);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr parseToPCLRGB(const rs2::points& points, const rs2::video_frame& color) {

    // Object Declaration (Point Cloud)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Declare Tuple for RGB value Storage (<t0>, <t1>, <t2>)
    std::tuple<uint8_t, uint8_t, uint8_t> RGB_Color;

    //================================
    // PCL Cloud Object Configuration
    //================================
    // Convert data captured from Realsense camera to Point Cloud
    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize(points.size());

    auto Texture_Coord = points.get_texture_coordinates();
    auto Vertex = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < points.size(); i++)
    {
        //===================================
        // Mapping Depth Coordinates
        // - Depth data stored as XYZ values
        //===================================
        cloud->points[i].x = Vertex[i].x;
        cloud->points[i].y = Vertex[i].y;
        cloud->points[i].z = Vertex[i].z;

        // Obtain color texture for specific point
        RGB_Color = getTextureRGB(color, Texture_Coord[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = std::get<2>(RGB_Color); // Reference tuple<2>
        cloud->points[i].g = std::get<1>(RGB_Color); // Reference tuple<1>
        cloud->points[i].b = std::get<0>(RGB_Color); // Reference tuple<0>

    }

    return cloud; // PCL RGB Point Cloud generated
}

pcl::PointCloud<pcl::PointXYZ>::Ptr parseToPCL(const rs2::points& points)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

void print4x4Matrix(const Eigen::Matrix4f& matrix)
{
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
    std::cout << std::endl;
}

float getDistanceToCenter()
{
    rs2::pipeline pipe;
    pipe.start();

    // Capturando 30 frames para autoexposición
    for (int i = 0; i < 30; i++) pipe.wait_for_frames();

    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    rs2::video_frame color = frames.get_color_frame();
    int height = depth.get_height();
    int width = depth.get_width();
    return depth.get_distance(width / 2, height / 2);
}

void adquisitionFormRealSense(bool haveToApplyPreprocessingFilters)
{
    rs2::decimation_filter decimation_filter;
    rs2::pointcloud pointcloud;
    rs2::pipeline pipe;
    pipe.start();

    std::cout << "Capturando 30 frames para autoexposicion..." << std::endl;
    for (int i = 0; i < 30; i++) pipe.wait_for_frames();

    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    rs2::video_frame color = frames.get_color_frame();

    if (haveToApplyPreprocessingFilters)
    {
        pointcloud.map_to(color);
        depth = decimation_filter.process(depth);
        auto points = pointcloud.calculate(depth);
        points.export_to_ply(DataPath + "capturedImage.ply", color);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPLYFile(DataPath + "capturedImage.ply", *cloud) < 0)
        {
            std::cerr << "Error loading cloud capturedImage.ply" << std::endl;
            exit(-1);
        }
        applyPreprocessingFilterTo(cloud);
        pcl::io::savePLYFile(DataPath + "capturedImage.ply", *cloud);
    }
    else
    {
        pointcloud.map_to(color);
        depth = decimation_filter.process(depth);
        auto points = pointcloud.calculate(depth);
        points.export_to_ply(DataPath + "capturedImage.ply", color);
    }
}

void _test_adquisitionFormRealSenseWithDifferentDecimationFilter(bool haveToApplyPreprocessingFilters)
{
    rs2::decimation_filter decimation_filter2(2); // default
    rs2::decimation_filter decimation_filter3(3);
    rs2::decimation_filter decimation_filter4(4);
    rs2::decimation_filter decimation_filter5(5);
    rs2::pointcloud pointcloud;
    rs2::pipeline pipe;
    pipe.start();

    std::cout << "Capturando 30 frames para autoexposicion..." << std::endl;
    for (int i = 0; i < 30; i++) pipe.wait_for_frames();

    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    rs2::video_frame color = frames.get_color_frame();

    if (haveToApplyPreprocessingFilters)
    {

    }
    else
    {
        pointcloud.map_to(color);
        auto depth2 = decimation_filter2.process(depth);
        auto depth3 = decimation_filter3.process(depth);
        auto depth4 = decimation_filter4.process(depth);
        auto depth5 = decimation_filter5.process(depth);
        auto points = pointcloud.calculate(depth);
        points.export_to_ply(DataPath + "capturedImage_decimationFilter_1.ply", color);
        points = pointcloud.calculate(depth2);
        points.export_to_ply(DataPath + "capturedImage_decimationFilter_2.ply", color);
        points = pointcloud.calculate(depth3);
        points.export_to_ply(DataPath + "capturedImage_decimationFilter_3.ply", color);
        points = pointcloud.calculate(depth4);
        points.export_to_ply(DataPath + "capturedImage_decimationFilter_4.ply", color);
        points = pointcloud.calculate(depth5);
        points.export_to_ply(DataPath + "capturedImage_decimationFilter_5.ply", color);
    }
}


void _test_adquisitionFromRealSenseLoopWithDifferentDecimationFilter(bool haveToApplyPreprocessingFilters)
{
    customMkdir(DataPath);
    customMkdir(DataPath + "adquisitionFromRealSense/");
    customMkdir(DataPath + "adquisitionFromRealSense/decimationFilter1/");
    customMkdir(DataPath + "adquisitionFromRealSense/decimationFilter2/");
    customMkdir(DataPath + "adquisitionFromRealSense/decimationFilter3/");
    customMkdir(DataPath + "adquisitionFromRealSense/decimationFilter4/");
    customMkdir(DataPath + "adquisitionFromRealSense/decimationFilter5/");
    customMkdir(DataPath + "preprocessedPointClouds/");
    customMkdir(DataPath + "preprocessedPointClouds/decimationFilter1/");
    customMkdir(DataPath + "preprocessedPointClouds/decimationFilter2/");
    customMkdir(DataPath + "preprocessedPointClouds/decimationFilter3/");
    customMkdir(DataPath + "preprocessedPointClouds/decimationFilter4/");
    customMkdir(DataPath + "preprocessedPointClouds/decimationFilter5/");

    std::vector<rs2::points> points1, points2, points3, points4, points5;
    std::vector<rs2::frame> colors, depths1, depths2, depths3, depths4, depths5;
    
    rs2::decimation_filter decimation_filter2(2); // default
    rs2::decimation_filter decimation_filter3(3);
    rs2::decimation_filter decimation_filter4(4);
    rs2::decimation_filter decimation_filter5(5);
    rs2::pointcloud pointcloud;
    rs2::pipeline pipe;
    pipe.start();

    std::cout << "Capturando 30 frames para autoexposicion..." << std::endl;
    for (auto i = 0; i < 30; i++) pipe.wait_for_frames();

    int i = 0;
    int time = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (time < RealSenseCaptureTotalTime_ms)
    {
        std::cout << i << " - Captura" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(RealSenseCaptureTimingBetweenCaptures_ms));
        auto fs = pipe.wait_for_frames();

        auto depth = fs.get_depth_frame();
        auto color = fs.get_color_frame();

        depth.keep();
        color.keep();

        depths1.push_back(depth);
        depths1[i].keep();
        colors.push_back(color);

        pointcloud.map_to(colors[i]);
        depths2.push_back(decimation_filter2.process(depth));
        depths2[i].keep();
        depths3.push_back(decimation_filter3.process(depth));
        depths3[i].keep();
        depths4.push_back(decimation_filter4.process(depth));
        depths4[i].keep();
        depths5.push_back(decimation_filter5.process(depth));
        depths5[i].keep();
        points1.push_back(pointcloud.calculate(depths1[i]));
        points1[i].keep();
        points2.push_back(pointcloud.calculate(depths2[i]));
        points2[i].keep();
        points3.push_back(pointcloud.calculate(depths3[i]));
        points3[i].keep();
        points4.push_back(pointcloud.calculate(depths4[i]));
        points4[i].keep();
        points5.push_back(pointcloud.calculate(depths5[i]));
        points5[i].keep();
        
        i++;
        auto stop = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    }

    if (haveToApplyPreprocessingFilters)
    {

    }
    else
    {
        for (i = 0; i < depths1.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths1[i].get_frame_number() << " como fichero con filtro diezmado 1" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/decimationFilter1/" << i << ".ply";
            points1[i].export_to_ply(filename.str(), colors[i]);
        }
        for (i = 0; i < depths2.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths2[i].get_frame_number() << " como fichero con filtro diezmado 2" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/decimationFilter2/" << i << ".ply";
            points2[i].export_to_ply(filename.str(), colors[i]);
        }
        for (i = 0; i < depths3.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths3[i].get_frame_number() << " como fichero con filtro diezmado 3" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/decimationFilter3/" << i << ".ply";
            points3[i].export_to_ply(filename.str(), colors[i]);
        }
        for (i = 0; i < depths4.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths4[i].get_frame_number() << " como fichero con filtro diezmado 4" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/decimationFilter4/" << i << ".ply";
            points4[i].export_to_ply(filename.str(), colors[i]);
        }
        for (i = 0; i < depths5.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths5[i].get_frame_number() << " como fichero con filtro diezmado 5" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/decimationFilter5/" << i << ".ply";
            points5[i].export_to_ply(filename.str(), colors[i]);
        }
    }
}

void adquisitionFromRealSenseLoop(bool haveToApplyPreprocessingFilters)
{
    customMkdir(DataPath);
    customMkdir(DataPath + "adquisitionFromRealSense/");
    customMkdir(DataPath + "preprocessedPointClouds/");

    std::vector<rs2::points> points;
    std::vector<rs2::frame> colors, depths;

    rs2::decimation_filter decimation_filter;
    rs2::pointcloud pointcloud;
    rs2::pipeline pipe;
    pipe.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout << "EMPIEZA" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));


    std::cout << "Capturando 30 frames para autoexposicion..." << std::endl;
    for (auto i = 0; i < 30; i++) pipe.wait_for_frames();

    int i = 0;
    int time = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (time < RealSenseCaptureTotalTime_ms)
    {
        std::cout << i << " - Captura" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(RealSenseCaptureTimingBetweenCaptures_ms));
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

        i++;
        auto stop = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    }

    if (haveToApplyPreprocessingFilters)
    {
        for (i = 0; i < depths.size(); i++)
        {
            std::cout << i << " - Convirtiendo frame " << depths[i].get_frame_number() << " a nube de puntos PCL" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/" << i << ".ply";

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in = parseToPCLRGB(points[i], colors[i]);
            applyPreprocessingFilterTo(cloud_in);
            pcl::io::savePLYFile(filename.str(), *cloud_in);
        }
    }
    else
    {
        for (i = 0; i < depths.size(); i++)
        {
            std::cout << i << " - Guardando frame " << depths[i].get_frame_number() << " como fichero" << std::endl;
            std::stringstream filename;
            filename << DataPath << "adquisitionFromRealSense/" << i << ".ply";
            points[i].export_to_ply(filename.str(), colors[i]);
        }
    }
}

double degreesToRadians(double degrees)
{
    return degrees * M_PI / 180;
}

void applyPreprocessingFilterTo(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
    //Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    //double theta = degreesToRadians(-22);
    //transformation_matrix(1, 1) = cos(theta);
    //transformation_matrix(1, 2) = -sin(theta);
    //transformation_matrix(2, 1) = sin(theta);
    //transformation_matrix(2, 2) = cos(theta);
    //pcl::transformPointCloud(*cloud, *cloud, transformation_matrix);
    //theta = degreesToRadians(-4);
    //transformation_matrix = Eigen::Matrix4f::Identity();
    //transformation_matrix(0, 0) = cos(theta);
    //transformation_matrix(0, 1) = -sin(theta);
    //transformation_matrix(1, 0) = sin(theta);
    //transformation_matrix(1, 1) = cos(theta);
    //pcl::transformPointCloud(*cloud, *cloud, transformation_matrix);

    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-ResizeCaptureLeft_m, ResizeCaptureRight_m);
    pass.filter(*cloud);
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-ResizeCaptureBottom_m, ResizeCaptureTop_m);
    pass.filter(*cloud);
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ResizeCenterDistance_m - ResizeCaptureBehind_m, ResizeCenterDistance_m + ResizeCaptureFront_m);
    pass.filter(*cloud);

    // Apply Statistical Outlier Removal Filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(60); // cuanto más alto mejor se mantienen los puntos del modelo
    sor.setStddevMulThresh(0.5);
    sor.filter(*cloud);
}

void applyPreprocessingFilterToRealSenseCaptures()
{
    customMkdir(DataPath);
    customMkdir(DataPath + "preprocessedPointClouds/");

    for (int i = 0; i < NumberOfCaptureFiles; i++) {
        std::stringstream filename;
        //filename << DataPath << "adquisitionFromRealSense/" << i << ".ply";
        filename << DataPath << "adquisitionFromRealSense/" << i << ".ply";

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPLYFile(filename.str(), *cloud_in) < 0)
        {
            std::cerr << "Error loading cloud " << filename.str() << std::endl;
            exit(-1);
        }

        applyPreprocessingFilterTo(cloud_in);

        filename.str(std::string());
        filename << DataPath << "preprocessedPointClouds/" << i << ".ply";
        pcl::io::savePLYFile(filename.str(), *cloud_in);
    }
}

Eigen::Matrix4f applyICP(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& targetPointCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& sourcePointCloud)
{
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(ICP_Iterations);
    icp.setMaxCorrespondenceDistance(ICP_MaxCorrespondenceDistance);
    icp.setInputSource(sourcePointCloud);
    icp.setInputTarget(targetPointCloud);
    icp.align(*sourcePointCloud);
    std::cout << "SCORE " <<icp.getFitnessScore();
    return icp.getFinalTransformation();
}

Eigen::Matrix4f applyCudaICP(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_cloud_target, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_cloud_source)
{
#ifdef _WINDOWS_
    return Eigen::Matrix4f::Identity();
#endif
#ifdef _LINUX_
    int nP = pcl_cloud_source->size();
    int nQ = pcl_cloud_target->size();
    float* nPdata = (float*)pcl_cloud_source->points.data();
    float* nQdata = (float*)pcl_cloud_target->points.data();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> time_span =
        std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

    Eigen::Matrix4f matrix_icp = Eigen::Matrix4f::Identity();
    //std::cout << "matrix_icp native value "<< std::endl;
    //print4x4Matrix(matrix_icp);
    void* cudaMatrix = NULL;
    cudaMatrix = malloc(sizeof(float) * 4 * 4);
    memset(cudaMatrix, 0, sizeof(float) * 4 * 4);
    std::cout << "------------checking CUDA ICP(GPU)---------------- " << std::endl;
    /************************************************/
    cudaStream_t stream = NULL;
    cudaStreamCreate(&stream);

    float* PUVM = NULL;
    cudaMallocManaged(&PUVM, sizeof(float) * 4 * nP, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, PUVM);
    cudaMemcpyAsync(PUVM, nPdata, sizeof(float) * 4 * nP, cudaMemcpyHostToDevice, stream);

    float* QUVM = NULL;
    cudaMallocManaged(&QUVM, sizeof(float) * 4 * nQ, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, QUVM);
    cudaMemcpyAsync(QUVM, nQdata, sizeof(float) * 4 * nQ, cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    cudaICP icpTest(nP, nQ, stream);

    t1 = std::chrono::steady_clock::now();
    icpTest.icp((float*)PUVM, nP, (float*)QUVM, nQ, ICP_Iterations, 1e-12, cudaMatrix, stream);
    t2 = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    std::cout << "CUDA ICP by Time: " << time_span.count() << " ms." << std::endl;
    cudaStreamDestroy(stream);
    /************************************************/
    memcpy(matrix_icp.data(), cudaMatrix, sizeof(float) * 4 * 4);
    std::cout << "CUDA ICP fitness_score: NI IDEA JULIO"; // << calculateFitneeScore( pcl_cloud_source, pcl_cloud_target, transformation_matrix) << std::endl;
    std::cout << "matrix_icp calculated Matrix by Class ICP " << std::endl;

    cudaFree(PUVM);
    cudaFree(QUVM);
    free(cudaMatrix);

    pcl::transformPointCloud(*pcl_cloud_target, *pcl_cloud_target, matrix_icp);
    return matrix_icp;
#endif
}

void pointCloudReconstructionUsingICP(bool haveToUseCudaImplementation, bool haveToSaveMiddleStepsForTesting)
{
    customMkdir(DataPath);
    customMkdir(DataPath + "outputICP/");
    customMkdir(DataPath + "outputICP_CUDA/");

    std::string path = haveToUseCudaImplementation ? "outputICP_CUDA/" : "outputICP/";

    // Initialize timer
    auto start = std::chrono::high_resolution_clock::now();    

    Eigen::Matrix4f initialTransformationMatrix;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::stringstream filename;
    filename << DataPath << "preprocessedPointClouds/" << 0 << ".ply";
    if (pcl::io::loadPLYFile(filename.str(), *targetPointCloud) < 0)
    {
        std::cerr << "Error loading cloud " << filename.str() << std::endl;
        exit(-1);
    }
    if (haveToSaveMiddleStepsForTesting)
    {
        filename.str(std::string());
        filename << DataPath << path << 0 << ".ply";
        pcl::io::savePLYFile(filename.str(), *targetPointCloud);
    }

    for (int i = 1; i < NumberOfCaptureFiles; i++)
    {
        // Read PLY objects
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        filename.str(std::string());
        filename << DataPath << "preprocessedPointClouds/" << i << ".ply";
        if (pcl::io::loadPLYFile(filename.str(), *sourcePointCloud) < 0)
        {
            std::cerr << "Error loading cloud " << filename.str() << std::endl;
            exit(-1);
        }

        std::cout << std::endl;
        std::cout << "----------------------------------------- " << i << " -----------------------------------------" << std::endl;
        std::cout << std::endl;

        if (i != 1)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointCloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*sourcePointCloud, *sourcePointCloud_transformed, initialTransformationMatrix);
            sourcePointCloud = sourcePointCloud_transformed;

            if (haveToSaveMiddleStepsForTesting)
            {
                filename.str(std::string());
                filename << DataPath << path << i << "_transformacionInicial.ply";
                pcl::io::savePLYFile(filename.str(), *sourcePointCloud);
            }

            std::cout << "La matriz de rotacion y translacion previa a aplicar es: " << std::endl;
            print4x4Matrix(initialTransformationMatrix);
        }

        Eigen::Matrix4f icpTransformationMatrix;
        if (haveToUseCudaImplementation)
        {
            icpTransformationMatrix = applyCudaICP(targetPointCloud, sourcePointCloud);
        }
        else
        {
            icpTransformationMatrix = applyICP(targetPointCloud, sourcePointCloud);
        }

        if (haveToSaveMiddleStepsForTesting)
        {
            filename.str(std::string());
            filename << DataPath << path << i << "_alineado.ply";
            pcl::io::savePLYFile(filename.str(), *sourcePointCloud);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << "Applied " << ICP_Iterations << " ICP iteration(s) in " << time / 1000.0 << "s" << std::endl;
        std::cout << "La matriz de convergencia es: " << std::endl;
        print4x4Matrix(icpTransformationMatrix);

        if (i == 1)
        {
            initialTransformationMatrix = icpTransformationMatrix;
        }
        else
        {
            initialTransformationMatrix = icpTransformationMatrix * initialTransformationMatrix;
        }

        std::cout << "La matriz inicial de transformacion para la siguiente iteracion es: " << std::endl;
        print4x4Matrix(initialTransformationMatrix);

        *targetPointCloud = (*targetPointCloud) + (*sourcePointCloud);

        if (haveToSaveMiddleStepsForTesting)
        {
            filename.str(std::string());
            filename << DataPath << path << i << "_acumulado.ply";
            pcl::io::savePLYFile(filename.str(), *targetPointCloud);
        }
    }

    filename.str(std::string());
    filename << DataPath << path << "final.ply";
    pcl::io::savePLYFile(filename.str(), *targetPointCloud);
}

void applyPostprocessingFilterToIcpResult()
{
    std::stringstream filename;
    filename.str(std::string());
    filename << DataPath << "outputICP/final.ply";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile(filename.str(), *cloud) < 0)
    {
        std::cerr << "Error loading cloud " << filename.str() << std::endl;
        exit(-1);
    }

    // Apply Statistical Outlier Removal Filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(30); // cuanto más alto mejor se mantienen los puntos del modelo
    sor.setStddevMulThresh(3);
    sor.filter(*cloud);

    filename.str(std::string());
    filename << DataPath << "final.ply";
    pcl::io::savePLYFile(filename.str(), *cloud);
    filename.str(std::string());
}

void customMkdir(std::string s)
{
#ifdef _WINDOWS_
    _mkdir(s.c_str());
#endif
#ifdef _LINUX_
    mkdir(s.c_str(), 0777);
#endif
}

void createDirectories()
{
    customMkdir(DataPath);
    customMkdir(DataPath + "adquisitionFromRealSense/");
    customMkdir(DataPath + "preprocessedPointClouds/");
    customMkdir(DataPath + "outputICP/");
    customMkdir(DataPath + "outputICP_CUDA/");
    customMkdir(DataPath + "postprocessedPointClouds/");
}

void readParametersFromJson()
{
    Json::Value parameters;
    std::ifstream parametersFile("parameters.json", std::ifstream::binary);

    if (!parametersFile.is_open())
    {
        std::cerr << "No se ha encontrado el fichero 'parameters.json'";
        exit(-1);
    }
    parametersFile >> parameters;

    if (parameters["DataPath"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'DataPath' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["RealSenseCaptureTotalTime_ms"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'RealSenseCaptureTotalTime_ms' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["RealSenseCaptureTimingBetweenCaptures_ms"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'RealSenseCaptureTimingBetweenCaptures_ms' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["NumberOfCaptureFiles"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'NumberOfCaptureFiles' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCenterDistance_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCenterDistance_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureBottom_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureBottom_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureTop_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureTop_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureLeft_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureLeft_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureRight_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureRight_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureBehind_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureBehind_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ResizeCaptureFront_m"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ResizeCaptureFront_m' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ICP_Iterations"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ICP_Iterations' en el fichero 'parameters.json'.";
        exit(-1);
    }
    if (parameters["ICP_MaxCorrespondenceDistance"].isNull())
    {
        std::cerr << "Es obligatorio establecer un valor para 'ICP_MaxCorrespondenceDistance' en el fichero 'parameters.json'.";
        exit(-1);
    }

    DataPath = parameters["DataPath"].asString();
    RealSenseCaptureTotalTime_ms = parameters["RealSenseCaptureTotalTime_ms"].asInt();
    RealSenseCaptureTimingBetweenCaptures_ms = parameters["RealSenseCaptureTimingBetweenCaptures_ms"].asInt();
    NumberOfCaptureFiles = parameters["NumberOfCaptureFiles"].asInt();
    ResizeCenterDistance_m = parameters["ResizeCenterDistance_m"].asFloat() * -1;
    ResizeCaptureBottom_m = parameters["ResizeCaptureBottom_m"].asFloat();
    ResizeCaptureTop_m = parameters["ResizeCaptureTop_m"].asFloat();
    ResizeCaptureLeft_m = parameters["ResizeCaptureLeft_m"].asFloat();
    ResizeCaptureRight_m = parameters["ResizeCaptureRight_m"].asFloat();
    ResizeCaptureBehind_m = parameters["ResizeCaptureBehind_m"].asFloat();
    ResizeCaptureFront_m = parameters["ResizeCaptureFront_m"].asFloat();
    ICP_Iterations = parameters["ICP_Iterations"].asInt();
    ICP_MaxCorrespondenceDistance = parameters["ICP_MaxCorrespondenceDistance"].asFloat();
}

void test2()
{
    rs2::decimation_filter decimation_filter;
    rs2::pointcloud pointcloud;
    rs2::pipeline pipe;
    pipe.start();

    std::cout << "Capturando 30 frames para autoexposicion..." << std::endl;
    for (int i = 0; i < 30; i++) pipe.wait_for_frames();

    int time = 0;
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Captura" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(RealSenseCaptureTimingBetweenCaptures_ms));
    auto fs = pipe.wait_for_frames();

    auto depth = fs.get_depth_frame();
    auto color = fs.get_color_frame();

    depth.keep();
    color.keep();

    pointcloud.map_to(color);
    depth = decimation_filter.process(depth);
    depth.keep();
    rs2::points points = pointcloud.calculate(depth);
    points.keep();

    auto stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "Guardando frame " << depth.get_frame_number() << std::endl;
    std::stringstream filename;
    filename << DataPath << "adquisitionFromRealSense/realsense.ply";
    points.export_to_ply(filename.str(), color);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in = parseToPCLRGB(points, color);
    filename.str(std::string());
    filename << DataPath << "adquisitionFromRealSense/pcl.ply";
    pcl::io::savePLYFile(filename.str(), *cloud_in);
}

int main()
{
    readParametersFromJson();
    createDirectories();

    std::string s = "";
    while (s != "0")
    {
        std::cout << "----- OPCIONES -----" << std::endl;
        std::cout << "[1] - Adquisicion de capturas con RealSense." << std::endl;
        std::cout << "[2] - Aplicar filtros de pre-procesamiento en las imagenes 3D capturadas." << std::endl;
        std::cout << "[3] - Reconstruccion 3D mediantes algoritmo ICP." << std::endl;
        std::cout << "[4] - Reconstruccion 3D mediantes algoritmo CUDA-ICP." << std::endl;
        std::cout << "[5] - Aplicar filtros de post-procesamiento al resultado." << std::endl;
        std::cout << "----- Otras -----" << std::endl;
        std::cout << "[6] - Ver distancia al centro." << std::endl;
        std::cout << "[7] - Realizar una captura 3D." << std::endl;
        std::cout << "[8] - Realizar una captura 3D con filtros de pre-procesamiento." << std::endl;
        std::cout << "[9] - Realizar una captura 3D con diferentes filtros de diezmado." << std::endl;
        std::cout << "[10] - Adquisicion de capturas con RealSense con diferentes filtros de diezmado." << std::endl;
        std::cout << "----- ----- -----" << std::endl;
        std::cout << "[0] - Salir." << std::endl;
        std::cout << "Opcion: ";

        std::cin >> s;
        std::cout << std::endl;

        if (s == "1")
        {
            adquisitionFromRealSenseLoop(false);
        }
        if (s == "2")
        {
            applyPreprocessingFilterToRealSenseCaptures();
        }
        if (s == "3")
        {
            pointCloudReconstructionUsingICP(false, true);
        }
        if (s == "4")
        {
            pointCloudReconstructionUsingICP(true, true);
        }
        if (s == "5")
        {
            applyPostprocessingFilterToIcpResult();
        }
        if (s == "6")
        {
            float distance = getDistanceToCenter();
            std::cout << "La distancia al centro es de " << distance << "m." << std::endl;
        }
        if (s == "7")
        {
            adquisitionFormRealSense(false);
        }
        if (s == "8")
        {
            adquisitionFormRealSense(true);
        }
        if (s == "9")
        {
            _test_adquisitionFormRealSenseWithDifferentDecimationFilter(false);
        }
        if (s == "10")
        {
            _test_adquisitionFromRealSenseLoopWithDifferentDecimationFilter(false);
        }
        std::cout << std::endl;
    }

	return 0;
}
