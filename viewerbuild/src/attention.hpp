/**
 * Copyright (C) 2017 Chalmers Revere
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 */

#ifndef CFSD18_SENSATION_ATTENTION_HPP
#define CFSD18_SENSATION_ATTENTION_HPP

#include <thread>
#include <Eigen/Dense>
#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"
#include "cone.hpp"
#include "drawer.hpp"
class Drawer;


class Attention {
 public:
  Attention(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session &od4, Drawer &drawer);
  Attention(Attention const &) = delete;
  Attention &operator=(Attention const &) = delete;
  ~Attention();
  void nextContainer(cluon::data::Envelope data);

 private:
  void setUp(std::map<std::string, std::string> commandlineArguments); 
  void SaveOneCPCPointNoIntensity(const int &pointIndex,const uint16_t &distance_integer, const double &azimuth, const double &verticalAngle);
  void SavePointCloud(opendlv::proxy::PointCloudReading pointCloud);
  void ConeDetection();
  std::vector<std::vector<uint32_t>> NNSegmentation(Eigen::MatrixXd &pointCloudConeROI, const double &connectDistanceThreshold);
  std::vector<std::vector<uint32_t>> FindConesFromObjects(Eigen::MatrixXd &pointCloudConeROI, std::vector<std::vector<uint32_t>> &objectIndexList, const double &minNumOfPointsForCone, const double &maxNumOfPointsForCone, const double &nearConeRadiusThreshold, const double &farConeRadiusThreshold, const double &zRangeThreshold);
  Eigen::MatrixXd ExtractConeROI(const double &xBoundary, const double &yBoundary, const double &groundLayerZ,  const double &coneHeight);
  double CalculateXYDistance(Eigen::MatrixXd &pointCloud, const uint32_t &index1, const uint32_t &index2);
  double CalculateConeRadius(Eigen::MatrixXd &potentialConePointCloud);
  double GetZRange(Eigen::MatrixXd &potentialConePointCloud);
  void SendingConesPositions(Eigen::MatrixXd &pointCloudConeROI, std::vector<std::vector<uint32_t>> &coneIndexList);
  Eigen::Vector3f Cartesian2Spherical(double &x, double &y, double &z);
  Eigen::MatrixXd RANSACRemoveGround(Eigen::MatrixXd);
  Eigen::MatrixXd RemoveDuplicates(Eigen::MatrixXd);


  // Define constants to decode CPC message
  const double START_V_ANGLE = -15.0; //For each azimuth there are 16 points with unique vertical angles from -15 to 15 degrees
  const double V_INCREMENT = 2.0; //The vertical angle increment for the 16 points with the same azimuth is 2 degrees
  const double START_V_ANGLE_32 = -30.67; //The starting angle for HDL-32E. Vertical angle ranges from -30.67 to 10.67 degress, with alternating increment 1.33 and 1.34
  const double V_INCREMENT_32_A = 1.33; //The first vertical angle increment for HDL-32E
  const double V_INCREMENT_32_B = 1.34; //The second vertical angle increment for HDL-32E

  // Constants for degree transformation
  const double DEG2RAD = 0.017453292522222; // PI/180.0
  const double RAD2DEG = 57.295779513082325; // 1.0 / DEG2RAD;
  cluon::OD4Session &m_od4;
  std::mutex m_cpcMutex;
  bool m_CPCReceived;//Set to true when the first compact point cloud is received

  // Class variables to save point cloud 
  Eigen::MatrixXd m_pointCloud;
  int m_pointIndex;
  // Define constants and thresolds forclustering algorithm
  double m_xBoundary;
  double m_yBoundary;
  double m_groundLayerZ;
  double m_coneHeight;
  double m_connectDistanceThreshold;
  double m_layerRangeThreshold;
  uint16_t m_minNumOfPointsForCone;
  uint16_t m_maxNumOfPointsForCone;
  double m_farConeRadiusThreshold;
  double m_nearConeRadiusThreshold;
  double m_zRangeThreshold;
  cluon::data::TimeStamp m_CPCReceivedLastTime;
  double m_algorithmTime;
  Eigen::MatrixXd m_generatedTestPointCloud;
  // RANSAC thresholds
  double m_inlierRangeThreshold;
  double m_inlierFoundTreshold;
  double m_ransacIterations;
  double m_dotThreshold;
  uint32_t m_senderStamp = 0;
  Eigen::MatrixXd m_lastBestPlane;
  Drawer& m_drawer;

  std::vector<std::pair<bool, Cone>> m_coneFrame = {};

  int m_validCones = 0;
  int m_count = 0;
  


};


#endif
