


#ifndef DRAWER_HPP
#define DRAWER_HPP
#include <pangolin/pangolin.h>
#include <Eigen/Dense>
#include "cone.hpp"




class Drawer{
    public:
        Drawer(std::map<std::string,std::string> commandlineArgs);
        void drawRawPoints();
        void drawROIPoints();
        void drawRANSACPoints();
        void drawCones();
        void setRawPoints(Eigen::MatrixXd rawPoints);
        void setROIPoints(Eigen::MatrixXd ROIPoints);
        void setRANSACPoints(Eigen::MatrixXd RANSACPoints);
        void setCones(std::vector<Cone> m_cones);

    private:
        std::mutex m_rawMutex = {};
        std::mutex m_ROIMutex = {};
        std::mutex m_RANSACMutex = {};
        std::mutex m_coneMutex = {};

        Eigen::MatrixXd m_rawPoints = {};
        Eigen::MatrixXd m_ROIPoints = {};
        Eigen::MatrixXd m_RANSACPoints = {};
        std::vector<Cone> m_cones = {};

};
#endif