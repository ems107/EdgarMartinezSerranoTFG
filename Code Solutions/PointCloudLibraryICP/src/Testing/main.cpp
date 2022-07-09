
#include <iostream>
#include <string>
#include <direct.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc


#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/median_filter.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;

void print4x4Matrix(const Eigen::Matrix4d& matrix)
{
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

int transformPointCloudWithIcpTest() {
    // The point clouds we will be using
    PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
    PointCloudT::Ptr cloud_out(new PointCloudT);  // ICP output point cloud
    int iterations = 200;  // Default number of ICP iterations

    pcl::console::TicToc time;
    time.tic();

    if (pcl::io::loadPLYFile(".data/test.ply", *cloud_in) < 0)
    {
        PCL_ERROR("Error loading cloud .\n");
        return (-1);
    }

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    double alpha = M_PI / 32;  // The angle of rotation in radians
    double theta = M_PI / -7;  // The angle of rotation in radians
    transformation_matrix(0, 0) = std::cos(alpha);
    transformation_matrix(0, 1) = -sin(alpha);
    transformation_matrix(1, 0) = sin(alpha);
    transformation_matrix(1, 1) = std::cos(alpha);
    transformation_matrix(1, 2) = -sin(theta);
    transformation_matrix(2, 1) = sin(theta);
    pcl::transformPointCloud(*cloud_in, *cloud_in, transformation_matrix);

    pcl::io::savePLYFile(".data/test_saved.ply", *cloud_in);
    return 0;

    if (pcl::io::loadPLYFile(".data/resizedPLYFiles/77.ply", *cloud_out) < 0)
    {
        PCL_ERROR("Error loading cloud .\n");
        return (-1);
    }

    // The Iterative Closest Point algorithm
    time.tic();
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(iterations);
    icp.setInputSource(cloud_out);
    icp.setInputTarget(cloud_in);
    icp.align(*cloud_out);
    std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;

    if (icp.hasConverged())
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
        print4x4Matrix(transformation_matrix);

        pcl::io::savePLYFile(".data/result.ply", *cloud_out);
    }
    else
    {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }

    return (0);
}

int transformPointCloudWithIcpRedrawing(int files) {
    // Default number of ICP iterations
    int iterations = 200;

    // Initialize timer
    pcl::console::TicToc time;
    time.tic();

    Eigen::Matrix4d finalTransformationMatrix;

    PointCloudT::Ptr cloud_in(new PointCloudT);
    std::stringstream filenameInput;
    filenameInput << ".data/resizedPLYFiles/" << 0 << ".ply";
    if (pcl::io::loadPLYFile(filenameInput.str(), *cloud_in) < 0)
    {
        PCL_ERROR("Error loading cloud .\n");
        return (-1);
    }
    std::stringstream filenameOutputFirstFile;
    filenameOutputFirstFile << ".data/outputIcpRedrawing/" << 0 << ".ply";
    pcl::io::savePLYFile(filenameOutputFirstFile.str(), *cloud_in);

    for (int i = 0; i < files - 1; i++) {
        // Read PLY objects
        PointCloudT::Ptr cloud_out(new PointCloudT);
        std::stringstream filenameOutput;
        filenameOutput << ".data/resizedPLYFiles/" << i + 1 << ".ply";
        if (pcl::io::loadPLYFile(filenameOutput.str(), *cloud_out) < 0)
        {
            PCL_ERROR("Error loading cloud .\n");
            return (-1);
        }

        std::cout << std::endl;
        std::cout << "----------------------------------------- " << i + 1 << " -----------------------------------------" << std::endl;
        std::cout << std::endl;

        if (i != 0) {
            PointCloudT::Ptr cloud_out_transformed(new PointCloudT);
            pcl::transformPointCloud(*cloud_out, *cloud_out_transformed, finalTransformationMatrix);
            cloud_out = cloud_out_transformed;

            std::stringstream filenameOutputTransformed;
            filenameOutputTransformed << ".data/outputIcpRedrawing/" << i + 1 << "_trans.ply";
            pcl::io::savePLYFile(filenameOutputTransformed.str(), *cloud_out);

            std::cout << "La matriz de rotación y translación previa a aplicar: " << std::endl;
            print4x4Matrix(finalTransformationMatrix);
        }

        // The Iterative Closest Point algorithm
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setMaximumIterations(iterations);
        icp.setMaxCorrespondenceDistance(0.005);
        icp.setInputSource(cloud_out);
        icp.setInputTarget(cloud_in);
        icp.align(*cloud_out);

        std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() / 1000 << "s" << std::endl;

        if (icp.hasConverged())
        {
            std::cout << "ICP has converged, score is " << icp.getFitnessScore() << std::endl;
            std::cout << std::endl;
            std::cout << "La matriz de convergencia es: " << std::endl;
            Eigen::Matrix4d transformationMatrix = icp.getFinalTransformation().cast<double>();
            print4x4Matrix(transformationMatrix);

            if (i == 0) {
                finalTransformationMatrix = transformationMatrix;
            }
            else {
                finalTransformationMatrix = transformationMatrix * finalTransformationMatrix;
            }

            *cloud_in = (*cloud_in) + (*cloud_out);

            // Apply Statistical Outlier Removal Filter
            pcl::StatisticalOutlierRemoval<PointT> sor;
            sor.setInputCloud(cloud_in);
            sor.setMeanK(50);
            sor.setStddevMulThresh(4);
            sor.filter(*cloud_in);

            std::stringstream filenameOutput;
            filenameOutput << ".data/outputIcpRedrawing/" << i + 1 << ".ply";
            pcl::io::savePLYFile(filenameOutput.str(), *cloud_in);
        }
        else
        {
            PCL_ERROR("\nICP has not converged.\n");
            return (-1);
        }
    }

    return (0);
}

int transformPointCloudWithIcp(int files) {
    // Default number of ICP iterations
    int iterations = 150;

    // Initialize timer
    pcl::console::TicToc time;
    time.tic();

    Eigen::Matrix4d finalTransformationMatrix;

    for (int i = 0; i < files-1; i++) {
        // Read PLY objects
        PointCloudT::Ptr cloud_in(new PointCloudT);
        std::stringstream filenameInput;
        filenameInput << ".data/resizedPLYFiles/" << i << ".ply";
        if (pcl::io::loadPLYFile(filenameInput.str(), *cloud_in) < 0)
        {
            PCL_ERROR("Error loading cloud .\n");
            return (-1);
        }
        PointCloudT::Ptr cloud_out(new PointCloudT);
        std::stringstream filenameOutput;
        filenameOutput << ".data/resizedPLYFiles/" << i + 1 << ".ply";
        if (pcl::io::loadPLYFile(filenameOutput.str(), *cloud_out) < 0)
        {
            PCL_ERROR("Error loading cloud .\n");
            return (-1);
        }

        // The Iterative Closest Point algorithm
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setMaximumIterations(iterations);
        icp.setMaxCorrespondenceDistance(0.01); // gato
        //icp.setMaxCorrespondenceDistance(0.03);
        icp.setInputSource(cloud_out);
        icp.setInputTarget(cloud_in);
        icp.align(*cloud_out);
        std::cout << i + 1 << " - Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;

        if (icp.hasConverged())
        {
            std::cout << "\nICP " << i + 1 << " has converged, score is " << icp.getFitnessScore() << std::endl;
            Eigen::Matrix4d transformationMatrix = icp.getFinalTransformation().cast<double>();
            print4x4Matrix(transformationMatrix);

            PointCloudT::Ptr cloud_out_transformed(new PointCloudT);

            if (i == 0) {
                cloud_out_transformed = cloud_out;
                finalTransformationMatrix = transformationMatrix;
            }
            else {
                pcl::transformPointCloud(*cloud_out, *cloud_out_transformed, finalTransformationMatrix);
                finalTransformationMatrix = transformationMatrix * finalTransformationMatrix;
            }

            std::stringstream filenameOutputTransformed;
            filenameOutputTransformed << ".data/outputIcp/" << i + 1 << ".ply";
            pcl::io::savePLYFile(filenameOutputTransformed.str(), *cloud_out_transformed);

            if (i == 0) {
                std::stringstream filenameOutputFirstFile;
                filenameOutputFirstFile << ".data/outputIcp/" << i << ".ply";
                pcl::io::savePLYFile(filenameOutputFirstFile.str(), *cloud_in);
            }
        }
        else
        {
            PCL_ERROR("\nICP has not converged.\n");
            return (-1);
        }
    }

    return (0);
}

int transformPointCloudWithIcpRecursive(int files) {
    // Default number of ICP iterations
    int iterations = 50;

    // Initialize timer
    pcl::console::TicToc time;
    time.tic();

    Eigen::Matrix4d finalTransformationMatrix;

    for (int i = 0; i < files - 1; i++) {
        if (i == 0) {
            PointCloudT::Ptr cloud_in(new PointCloudT);
            std::stringstream filenameInput;
            filenameInput << ".data/resizedPLYFiles/" << i << ".ply";
            if (pcl::io::loadPLYFile(filenameInput.str(), *cloud_in) < 0)
            {
                PCL_ERROR("Error loading cloud .\n");
                return (-1);
            }
            std::stringstream filenameOutputFirstFile;
            filenameOutputFirstFile << ".data/outputIcpRecursive/" << i << ".ply";
            pcl::io::savePLYFile(filenameOutputFirstFile.str(), *cloud_in);
        }

        PointCloudT::Ptr cloud_out(new PointCloudT);
        std::stringstream filenameOutput;
        filenameOutput << ".data/resizedPLYFiles/" << i + 1 << ".ply";
        if (pcl::io::loadPLYFile(filenameOutput.str(), *cloud_out) < 0)
        {
            PCL_ERROR("Error loading cloud .\n");
            return (-1);
        }

        for (int j = i; j >= 0; j--) {
            // Read PLY objects
            PointCloudT::Ptr cloud_in(new PointCloudT);
            std::stringstream filenameInput;
            filenameInput << ".data/resizedPLYFiles/" << j << ".ply";
            if (pcl::io::loadPLYFile(filenameInput.str(), *cloud_in) < 0)
            {
                PCL_ERROR("Error loading cloud .\n");
                return (-1);
            }

            // The Iterative Closest Point algorithm
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setMaximumIterations(iterations);
            icp.setMaxCorrespondenceDistance(0.02);
            icp.setInputSource(cloud_out);
            icp.setInputTarget(cloud_in);
            icp.align(*cloud_out);
            std::cout << i + 1 << " with cloud " << j << " - Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;

            if (icp.hasConverged())
            {
                std::cout << "\nICP " << i + 1 << " has converged, score is " << icp.getFitnessScore() << std::endl;
                Eigen::Matrix4d transformationMatrix = icp.getFinalTransformation().cast<double>();
                print4x4Matrix(transformationMatrix);
            }
            else
            {
                PCL_ERROR("\nICP has not converged.\n");
                return (-1);
            }
        }

        std::cout << i + 1 << " finished in " << time.toc() << " ms" << std::endl;
        std::stringstream filenameOutputTransformed;
        filenameOutputTransformed << ".data/outputIcpRecursive/" << i + 1 << ".ply";
        pcl::io::savePLYFile(filenameOutputTransformed.str(), *cloud_out);
    }

    return (0);
}

int resizePLYFiles(int files, float centerDistance, float bottom, float top, float left, float right, float behind, float front) {
    for (int i = 0; i < files; i++) {
        std::stringstream inputFilename;
        inputFilename << ".data/inputRealsensePLY/" << i << ".ply";

        PointCloudT::Ptr cloud_in(new PointCloudT);
        if (pcl::io::loadPLYFile(inputFilename.str(), *cloud_in) < 0)
        {
            PCL_ERROR("Error loading cloud .\n");
            return (-1);
        }

        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
        double alpha = M_PI / 32;  // The angle of rotation in radians
        double theta = M_PI / -7;  // The angle of rotation in radians
        //transformation_matrix(0, 0) = std::cos(alpha);
        //transformation_matrix(0, 1) = -sin(alpha);
        //transformation_matrix(1, 0) = sin(alpha);
        //transformation_matrix(1, 1) = std::cos(alpha);
        transformation_matrix(1, 2) = -sin(theta);
        transformation_matrix(2, 1) = sin(theta);
        pcl::transformPointCloud(*cloud_in, *cloud_in, transformation_matrix);

        std::vector<int> pointToDelete;
        for (int i = 0; i < cloud_in->points.size(); i++) {
            if (cloud_in->points[i].z < centerDistance - behind ||
                cloud_in->points[i].z > centerDistance + front ||
                cloud_in->points[i].x < -left ||
                cloud_in->points[i].x > right ||
                cloud_in->points[i].y < -bottom ||
                cloud_in->points[i].y > top
                ) {
                pointToDelete.push_back(i);
            }
        }
        for (int i = pointToDelete.size(); i > 0; i--) {
            cloud_in->erase(cloud_in->points.begin() + pointToDelete[i-1]);
        }

        std::stringstream outputFilename;
        outputFilename << ".data/resizedPLYFiles/" << i << ".ply";

        // Apply Statistical Outlier Removal Filter
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud_in);
        sor.setMeanK(50);
        sor.setStddevMulThresh(2);
        sor.filter(*cloud_in);

        //// Apply Median Filter
        //pcl::MedianFilter<PointT> mf;
        //mf.setInputCloud(cloud_in);
        //mf.setWindowSize(50);
        //mf.setMaxAllowedMovement(1);
        //mf.applyFilter(*cloud_in);

        pcl::io::savePLYFile(outputFilename.str(), *cloud_in);
    }

    return 0;
}

int main(int argc, char* argv[])
{
    _mkdir(".data");
    _mkdir(".data/inputRealsensePLY");
    _mkdir(".data/outputIcp");
    _mkdir(".data/outputIcpRecursive");
    _mkdir(".data/outputIcpRedrawing");
    _mkdir(".data/resizedPLYFiles");

    // Params
    int files = 54;
    float centerDistance = -0.57;
    float bottom = 0.33;
    float top = 0.1;
    float left = 0.2;
    float right = 0.2;
    float behind = 0.20;
    float front = 0.20;

    return transformPointCloudWithIcp(files);
    return transformPointCloudWithIcpRedrawing(files);
    return resizePLYFiles(files, centerDistance, bottom, top, left, right, behind, front);
    return transformPointCloudWithIcpTest();
    return transformPointCloudWithIcpRecursive(files);
}