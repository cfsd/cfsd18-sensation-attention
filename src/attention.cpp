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

#include <iostream>
#include <cmath>
#include <vector>

#include "attention.hpp"

Attention::Attention(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session &od4) :
    m_od4(od4)
  , m_cpcMutex()
  , m_stateMachineMutex()
  , m_CPCReceived(false)
  , m_pointCloud()
  , m_pointIndex(0)
  , m_xBoundary()
  , m_yBoundary()
  , m_groundLayerZ()
  , m_coneHeight()
  , m_connectDistanceThreshold()
  , m_layerRangeThreshold()
  , m_minNumOfPointsForCone()
  , m_maxNumOfPointsForCone()
  , m_farConeRadiusThreshold()
  , m_nearConeRadiusThreshold()
  , m_zRangeThreshold()
  , m_CPCReceivedLastTime()
  , m_algorithmTime()
  , m_generatedTestPointCloud()
  , m_inlierRangeThreshold()
  , m_inlierFoundTreshold()
  , m_ransacIterations()
  , m_dotThreshold()
  , m_lastBestPlane()
{
  setUp(commandlineArguments);
}

Attention::~Attention()
{
}

void Attention::nextContainer(cluon::data::Envelope data)
{
  cluon::data::TimeStamp incommingDataTime = data.sampleTimeStamp();
  double timeSinceLastReceive = fabs(static_cast<double>(cluon::time::deltaInMicroseconds(incommingDataTime, m_CPCReceivedLastTime)));//fabs(static_cast<double>(incommingDataTime.microseconds()-m_CPCReceivedLastTime.microseconds())/1000000.0);

  std::lock_guard<std::mutex> lockStateMachine(m_stateMachineMutex);
  //std::cout << "Time since between 2 incomming messages: " << timeSinceLastReceive << "s" << std::endl;
  if (timeSinceLastReceive>m_algorithmTime && m_readyStateMachine){
    if(data.dataType() == opendlv::proxy::PointCloudReading::ID()) {
      cluon::data::TimeStamp TimeBeforeAlgorithm;
      m_CPCReceived = true;
      cluon::data::TimeStamp ts = data.sampleTimeStamp();
      m_CPCReceivedLastTime = ts;
      {
        std::lock_guard<std::mutex> lockCPC(m_cpcMutex);
        opendlv::proxy::PointCloudReading pointCloud = cluon::extractMessage<opendlv::proxy::PointCloudReading>(std::move(data));
	     TimeBeforeAlgorithm = cluon::time::convert(std::chrono::system_clock::now());
        SavePointCloud(pointCloud);
      }
	    //cluon::data::TimeStamp TimeBeforeAlgorithm = cluon::time::convert(std::chrono::system_clock::now());
      if(m_pointCloud.rows() > 20000){
        ConeDetection();
      }else{
        std::cout << "Point cloud size not sufficient, size is: " << m_pointCloud.rows() << std::endl;
      }
      cluon::data::TimeStamp TimeAfterAlgorithm = cluon::time::convert(std::chrono::system_clock::now());
      //std::cout << "before: " << TimeBeforeAlgorithm.microseconds() << "after: " << TimeAfterAlgorithm.microseconds() << std::endl;
      //double timeForProcessingOneScan = static_cast<double>(TimeAfterAlgorithm.microseconds()-TimeBeforeAlgorithm.microseconds())/1000000.0;
     // m_algorithmTime = timeForProcessingOneScan;

      m_algorithmTime = fabs(static_cast<double>(cluon::time::deltaInMicroseconds(TimeAfterAlgorithm, TimeBeforeAlgorithm)));
      std::cout << "Time for processing one scan of data is: " << m_algorithmTime/1000000 << "s" << std::endl;
    }
  }
}

void Attention::setUp(std::map<std::string, std::string> commandlineArguments)
{
  //m_startAngle = kv.getValue<double>("logic-cfsd18-sensation-attention.startAngle");
  //m_endAngle = kv.getValue<double>("logic-cfsd18-sensation-attention.endAngle");
  m_xBoundary = std::stod(commandlineArguments["xBoundary"]);
  m_yBoundary = std::stod(commandlineArguments["yBoundary"]);;
  m_groundLayerZ = std::stod(commandlineArguments["groundLayerZ"]);
  m_coneHeight = std::stod(commandlineArguments["coneHeight"]);
  m_connectDistanceThreshold = std::stod(commandlineArguments["connectDistanceThreshold"]);
  m_layerRangeThreshold = std::stod(commandlineArguments["layerRangeThreshold"]);
  m_minNumOfPointsForCone = std::stoi(commandlineArguments["minNumOfPointsForCone"]);
  m_maxNumOfPointsForCone = std::stoi(commandlineArguments["maxNumOfPointsForCone"]);
  m_farConeRadiusThreshold = std::stod(commandlineArguments["farConeRadiusThreshold"]);
  m_nearConeRadiusThreshold = std::stod(commandlineArguments["nearConeRadiusThreshold"]);
  m_zRangeThreshold = std::stod(commandlineArguments["zRangeThreshold"]);
  m_inlierRangeThreshold = std::stod(commandlineArguments["inlierRangeThreshold"]);
  m_inlierFoundTreshold = std::stod(commandlineArguments["inlierFoundThreshold"]);
  m_ransacIterations = std::stod(commandlineArguments["numberOfIterations"]);
  m_dotThreshold = std::stod(commandlineArguments["dotThreshold"]);
  m_senderStamp = std::stoi(commandlineArguments["id"]);
  m_lastBestPlane = Eigen::MatrixXd::Zero(1,4);
  m_lastBestPlane << 0,0,1,0;
  //ConeDetection();

}

void Attention::setReadyState(bool state){
  m_readyState = state;
}

void Attention::setStateMachineStatus(cluon::data::Envelope data){
  std::lock_guard<std::mutex> lockStateMachine(m_stateMachineMutex);
  auto machineStatus = cluon::extractMessage<opendlv::proxy::SwitchStateReading>(std::move(data));
  int state = machineStatus.state();
  if(state == 2){
    m_readyStateMachine = true;
  }
  
}

void Attention::SaveOneCPCPointNoIntensity(const int &pointIndex,const uint16_t &distance_integer, const double &azimuth, const double &verticalAngle)
{
  
  //Recordings before 2017 do not call hton() while storing CPC.
  //Hence, we only call ntoh() for recordings from 2017.
  uint16_t distanceCPCPoint = ntohs(distance_integer);
  double distance = 0.0;
  distance = static_cast<double>(distanceCPCPoint / 100.0f); //convert to meter from resolution 1 cm, only 1 cm is supported

  // Compute x, y, z coordinate based on distance, azimuth, and vertical angle
  double xyDistance = distance * cos(verticalAngle * static_cast<double>(DEG2RAD));
  double xData = xyDistance * sin(azimuth * static_cast<double>(DEG2RAD));
  double yData = xyDistance * cos(azimuth * static_cast<double>(DEG2RAD));
  double zData = distance * sin(verticalAngle * static_cast<double>(DEG2RAD));
  m_pointCloud.row(pointIndex) << xData,yData,zData;
}


void Attention::SavePointCloud(opendlv::proxy::PointCloudReading pointCloud){
  if (m_CPCReceived) {
    const double startAzimuth = pointCloud.startAzimuth();
    const double endAzimuth = pointCloud.endAzimuth();
    //std::cout << "Start Azimut: " << startAzimuth << "End Azimuth: " << endAzimuth << std::endl;
    const uint8_t entriesPerAzimuth = pointCloud.entriesPerAzimuth(); // numberOfLayers

    uint16_t distance_integer = 0;
    if (entriesPerAzimuth == 16) {//A VLP-16 CPC
      double azimuth = startAzimuth;
      const std::string distances = pointCloud.distances();
      const uint32_t numberOfPoints = distances.size() / 2;
      m_numberOfAzimuths = numberOfPoints / entriesPerAzimuth;
      const double azimuthIncrement = (endAzimuth - startAzimuth) / m_numberOfAzimuths;//Calculate the azimuth increment
      std::stringstream sstr(distances);
      m_pointCloud = Eigen::MatrixXd::Zero(numberOfPoints,3);
      m_pointIndex = 0;
      for (uint32_t azimuthIndex = 0; azimuthIndex < m_numberOfAzimuths; azimuthIndex++) {
          double verticalAngle = START_V_ANGLE;
          for (uint8_t sensorIndex = 0; sensorIndex < entriesPerAzimuth; sensorIndex++) {
              sstr.read((char*)(&distance_integer), 2); // Read distance value from the string in a CPC container point by point
              SaveOneCPCPointNoIntensity(m_pointIndex,distance_integer, azimuth, verticalAngle);
              m_pointIndex++;
              verticalAngle += V_INCREMENT;
          }
        azimuth += azimuthIncrement;
      }
    }
    //std::cout << "number of points are:"<< m_pointCloud.rows() << std::endl;
  } 
}

void Attention::ConeDetection(){
  //m_generatedTestPointCloud = GenerateTestPointCloud();
  cluon::data::TimeStamp startTime = cluon::time::convert(std::chrono::system_clock::now());


  
 
  cluon::data::TimeStamp processTime = cluon::time::convert(std::chrono::system_clock::now());
  //double timeElapsed = fabs(static_cast<double>(processTime.microseconds()-startTime.microseconds())/1000000.0);
  //std::cout << "Time elapsed for Extract RoI: " << timeElapsed << std::endl;

  //std::cout << "RANSAC" << std::endl;
  //std::cout << "number of points after ROI are:"<< pointCloudConeROI.rows() << std::endl;
  

  //Eigen::MatrixXd pcRefit = RANSACRemoveGround(pointCloudConeROI);

  Eigen::MatrixXd pcNoGround = layerRemoveGround(m_pointCloud);
  {
    std::lock_guard<std::mutex> lock(m_drawerMutex);
    m_pcAfterRANSAC = pcNoGround;
  }
  //std::cout << "RANSACSIZE: " << pcRefit.rows() << std::endl;

  Eigen::MatrixXd pcRefit = ExtractConeROI(pcNoGround,m_xBoundary, m_yBoundary, m_groundLayerZ, m_coneHeight);

   {
    std::lock_guard<std::mutex> lock(m_drawerMutex);
    m_pointCloudROI = pcRefit;
  }
  startTime = processTime;
  std::vector<int32_t> notFloorIndex;

  cluon::data::TimeStamp processTime2 = cluon::time::convert(std::chrono::system_clock::now());
  
  //timeElapsed = fabs(static_cast<double>(processTime2.microseconds()-startTime.microseconds())/1000000.0);
  //std::cout << "Time elapsed for RANSAC: " << timeElapsed << std::endl;
  startTime = processTime2;

  double numOfPointsAfterRANSAC = pcRefit.rows();
  if (numOfPointsAfterRANSAC <= 1000.0 && numOfPointsAfterRANSAC > 0)
  {
    std::vector<std::vector<uint32_t>> objectIndexList = NNSegmentation(pcRefit, m_connectDistanceThreshold); //out from ransac pointCloudConeROI to pointCloudFilt
    cluon::data::TimeStamp processTime3 = cluon::time::convert(std::chrono::system_clock::now());
    //timeElapsed = fabs(static_cast<double>(processTime3.microseconds()-startTime.microseconds())/1000000.0);
    //std::cout << "Time elapsed for NNSegmentation: " << timeElapsed << std::endl;
    startTime = processTime3;
    std::vector<std::vector<uint32_t>> coneIndexList = FindConesFromObjects(pcRefit, objectIndexList, m_minNumOfPointsForCone, m_maxNumOfPointsForCone, m_nearConeRadiusThreshold, m_farConeRadiusThreshold, m_zRangeThreshold);
    cluon::data::TimeStamp processTime4 = cluon::time::convert(std::chrono::system_clock::now());
    //timeElapsed = fabs(static_cast<double>(processTime4.microseconds()-startTime.microseconds())/1000000.0);
    //std::cout << "Time elapsed for Cone Detection: " << timeElapsed << std::endl;
    startTime = processTime4;
    //std::cout << "Number of Cones is: " << coneIndexList.size() << std::endl;
    SendingConesPositions(pcRefit, coneIndexList);


  }

}

std::vector<std::vector<uint32_t>> Attention::NNSegmentation(Eigen::MatrixXd &pointCloudConeROI, const double &connectDistanceThreshold){
  uint32_t numberOfPointConeROI = pointCloudConeROI.rows();
  //std::cout << "pc in NN: " << pointCloudConeROI.rows() << std::endl;
  std::vector<uint32_t> restPointsList(numberOfPointConeROI);
  for (uint32_t i = 0; i < numberOfPointConeROI; i++)
  {
    restPointsList[i] = i;
  }
  std::vector<std::vector<uint32_t>> objectIndexList;
  std::vector<uint32_t> tmpObjectIndexList; tmpObjectIndexList.push_back(restPointsList[0]);
  uint32_t tmpPointIndex = restPointsList[0];
  uint32_t positionOfTmpPointIndexInList = 0;
  uint32_t tmpPointIndexNext = 0;
  restPointsList.erase(restPointsList.begin());

  while (!restPointsList.empty())
  {
    uint32_t numberOfRestPoints = restPointsList.size();
    double minDistance = 100000; // assign a large value for inilization
    for (uint32_t i = 0; i < numberOfRestPoints; i++)
    {
      double distance = CalculateXYDistance(pointCloudConeROI, tmpPointIndex, restPointsList[i]);
      //std::cout << "Distance is " << distance << std::endl;
      if (distance < minDistance)
      {
        tmpPointIndexNext = restPointsList[i];
        positionOfTmpPointIndexInList = i;
        minDistance = distance;
      }

    }
    tmpPointIndex = tmpPointIndexNext;
    //std::cout << "Minimum Distance is " << minDistance << std::endl;
    // now we have minDistance and tmpPointIndex for next iteration
    if (minDistance <= connectDistanceThreshold)
    {
      tmpObjectIndexList.push_back(tmpPointIndex);
    } else {
      if (!tmpObjectIndexList.empty())
      {
        objectIndexList.push_back(tmpObjectIndexList);
      }
      tmpObjectIndexList.clear();
      tmpObjectIndexList.push_back(tmpPointIndex);
    }
    restPointsList.erase(restPointsList.begin()+positionOfTmpPointIndexInList);
  }
  if (!tmpObjectIndexList.empty())
  {
    objectIndexList.push_back(tmpObjectIndexList);
  }
  return objectIndexList;
}

std::vector<std::vector<uint32_t>> Attention::FindConesFromObjects(Eigen::MatrixXd &pointCloudConeROI, std::vector<std::vector<uint32_t>> &objectIndexList, const double &minNumOfPointsForCone, const double &maxNumOfPointsForCone, const double &nearConeRadiusThreshold, const double &farConeRadiusThreshold, const double &zRangeThreshold)
{
  uint32_t numberOfObjects = objectIndexList.size();
  //std::cout << "pc in findcones: " << pointCloudConeROI.rows() << std::endl;
  // Select those objects with reasonable number of points and save the object list in a new vector
  std::vector<std::vector<uint32_t>> objectIndexListWithNumOfPointsLimit;
  for (uint32_t i = 0; i < numberOfObjects; i ++)
  {
    std::vector<uint32_t> objectIndex = objectIndexList[i];
    uint32_t numberOfPointsOnObject = objectIndex.size();
    
  //std::cout << "object: "<< i << " : points: "<< numberOfPointsOnObject << numberOfObjects << std::endl;
    bool numberOfPointsLimitation = ((numberOfPointsOnObject >= minNumOfPointsForCone) && (numberOfPointsOnObject <= maxNumOfPointsForCone));
    if (numberOfPointsLimitation)
    {
      objectIndexListWithNumOfPointsLimit.push_back(objectIndex);
    }
  }

  // Select among previous potention cones with reasonable radius
  std::vector<std::vector<uint32_t>> coneIndexList;
  for (uint32_t i = 0; i < objectIndexListWithNumOfPointsLimit.size(); i ++)
  {
    std::vector<uint32_t> selectedObjectIndex = objectIndexListWithNumOfPointsLimit[i];
    uint32_t numberOfPointsOnSelectedObject = selectedObjectIndex.size();
    Eigen::MatrixXd potentialConePointCloud = Eigen::MatrixXd::Zero(numberOfPointsOnSelectedObject,3);
    for (uint32_t j = 0; j < numberOfPointsOnSelectedObject; j++)
    {
      potentialConePointCloud.row(j) << pointCloudConeROI(selectedObjectIndex[j],0),pointCloudConeROI(selectedObjectIndex[j],1),pointCloudConeROI(selectedObjectIndex[j],2);
    }

    double coneRadius = CalculateConeRadius(potentialConePointCloud);
    double zRange = GetZRange(potentialConePointCloud);
    //cout << "Cone size: " << coneRadius << endl;
    bool condition1 = (coneRadius < farConeRadiusThreshold); //Far point cones
    bool condition2 = (coneRadius>= farConeRadiusThreshold && coneRadius <= nearConeRadiusThreshold);
    bool condition3 = (zRange >= zRangeThreshold);  // Near point cones have to cover a larger Z range
    if (condition1 || (condition2 && condition3))
    {
      coneIndexList.push_back(selectedObjectIndex);
      
    }

  }

  return coneIndexList;

}

double Attention::CalculateConeRadius(Eigen::MatrixXd &potentialConePointCloud)
{
  double coneRadius = 0;
  uint32_t numberOfPointsOnSelectedObject = potentialConePointCloud.rows();
  double xMean = potentialConePointCloud.colwise().sum()[0]/numberOfPointsOnSelectedObject;
  double yMean = potentialConePointCloud.colwise().sum()[1]/numberOfPointsOnSelectedObject;
  for (uint32_t i = 0; i < numberOfPointsOnSelectedObject; i++)
  {
    double radius = sqrt(pow((potentialConePointCloud(i,0)-xMean),2)+pow((potentialConePointCloud(i,1)-yMean),2));
    if (radius >= coneRadius)
    {
      coneRadius = radius;
    }
  }
  return coneRadius;

}

double Attention::GetZRange(Eigen::MatrixXd &potentialConePointCloud)
{
  double zRange = potentialConePointCloud.colwise().maxCoeff()[2]-potentialConePointCloud.colwise().minCoeff()[2];
  return zRange;
}


Eigen::MatrixXd Attention::ExtractConeROI(Eigen::MatrixXd pointCloudFromGM, const double &xBoundary, const double &yBoundary, const double &groundLayerZ,  const double &coneHeight){
  uint32_t numberOfPointsCPC = pointCloudFromGM.rows();
  uint32_t numberOfPointConeROI = 0;
  std::vector<int> pointIndexConeROI;
  for (uint32_t i = 0; i < numberOfPointsCPC; i++)
  {
    if ((pointCloudFromGM(i,0) >= -xBoundary) && (pointCloudFromGM(i,0) <= xBoundary) && (pointCloudFromGM(i,1) <= yBoundary) && (pointCloudFromGM(i,1) >= 0.2) && (pointCloudFromGM(i,2) <= groundLayerZ + coneHeight))
    {
      pointIndexConeROI.push_back(i);
      numberOfPointConeROI ++;
    }
  }
  Eigen::MatrixXd pointCloudConeROI = Eigen::MatrixXd::Zero(numberOfPointConeROI,3);
  for (uint32_t j = 0; j < numberOfPointConeROI; j++)
  {
    pointCloudConeROI.row(j) << pointCloudFromGM(pointIndexConeROI[j],0),pointCloudFromGM(pointIndexConeROI[j],1),pointCloudFromGM(pointIndexConeROI[j],2);
  }

  return pointCloudConeROI;
}

double Attention::CalculateXYDistance(Eigen::MatrixXd &pointCloud, const uint32_t &index1, const uint32_t &index2)
{
  double x1 = pointCloud(index1,0);
  double y1 = pointCloud(index1,1);
  double x2 = pointCloud(index2,0);
  double y2 = pointCloud(index2,1);
  double distance = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
  return distance;
}

void Attention::SendingConesPositions(Eigen::MatrixXd &pointCloudConeROI, std::vector<std::vector<uint32_t>> &coneIndexList)
{
  int m = 0;
  double posShiftX = 0;
  double posShiftY = 0;
  uint32_t numberOfCones = coneIndexList.size();
  std::vector<uint32_t> removeIndex;
  //Eigen::MatrixXd conePoints = Eigen::MatrixXd::Zero(numberOfCones,3);
  m_validCones = 0;
  for(uint32_t i1 = 0; i1 < m_coneFrame.size(); i1++){
    if(m_coneFrame[i1].second.isValid()){
      m_validCones++;
    }
  }  
  if(m_validCones == 0){
      m_coneFrame.clear();
  }
  for (uint32_t i = 0; i < numberOfCones; i++)
  {
    uint32_t numberOfPointsOnCone = coneIndexList[i].size();
    double conePositionX = 0;
    double conePositionY = 0;
    double conePositionZ = 0;
    for (uint32_t j = 0; j< numberOfPointsOnCone; j++)
    {
      conePositionX += pointCloudConeROI(coneIndexList[i][j],0);
      conePositionY += pointCloudConeROI(coneIndexList[i][j],1);
      conePositionZ += pointCloudConeROI(coneIndexList[i][j],2);
    }
    conePositionX = conePositionX / numberOfPointsOnCone;
    conePositionY = conePositionY / numberOfPointsOnCone;
    conePositionZ = conePositionZ / numberOfPointsOnCone;
    //conePoints.row(i) << conePositionX,conePositionY,conePositionZ;

    Cone objectToValidate = Cone(conePositionX,conePositionY,conePositionZ);
    std::pair<bool, Cone> objectPair = std::pair<bool, Cone>(false,objectToValidate);

    if(m_validCones == 0){
        std::cout << "in empty frame" << std::endl;
        //m_coneFrame.clear();
        m_coneFrame.push_back(objectPair);
    }else{ 
      int k = 0;
      int frameSize = m_coneFrame.size();
      bool foundMatch = false;
      if(m_coneFrame[k].second.isValid()){
        if(objectPair.second.isThisMe(m_coneFrame[k].second.getX(),m_coneFrame[k].second.getY())){
          posShiftX += m_coneFrame[k].second.getX() - objectPair.second.getX();
          posShiftY += m_coneFrame[k].second.getY() - objectPair.second.getY();
          m++;
          //std::cout << "found match" << std::endl;
          m_coneFrame[k].second.setX(objectPair.second.getX());
          m_coneFrame[k].second.setY(objectPair.second.getY());
          m_coneFrame[k].second.addHit();
          m_coneFrame[k].first = true;
          foundMatch = true;
        }
      }
      while( k < frameSize && !foundMatch){
        if(!m_coneFrame[k].second.isValid()){
          k++;
          continue;
        }
        if(!m_coneFrame[k].first && objectPair.second.isThisMe(m_coneFrame[k].second.getX(),m_coneFrame[k].second.getY())){
          posShiftX += m_coneFrame[k].second.getX() - objectPair.second.getX();
          posShiftY += m_coneFrame[k].second.getY() - objectPair.second.getY();
          m++;
          //std::cout << "found match" << std::endl;
          m_coneFrame[k].second.setX(objectPair.second.getX());
          m_coneFrame[k].second.setY(objectPair.second.getY());
          m_coneFrame[k].second.addHit();
          m_coneFrame[k].first = true;
          foundMatch = true;  
        }
        k++;
      }
      if(!foundMatch){
        m_coneFrame.push_back(objectPair);
      }
    }   //std::cout << "Cone Sent" << std::endl;//std::cout << "a point sent out with distance: " <<conePoint.getDistance() <<"; azimuthAngle: " << conePoint.getAzimuthAngle() << "; and zenithAngle: " << conePoint.getZenithAngle() << std::endl;
  }
  std::lock_guard<std::mutex> lock(m_drawerMutex);
  m_sentCones.clear();
  int sentCounter = 0;
  for(uint32_t i = 0; i < m_coneFrame.size(); i++){
    if(m_coneFrame[i].second.isValid()){
      if(m_coneFrame[i].second.shouldBeRemoved()){
        //m_coneFrame.erase(m_coneFrame.begin()+i)  DO NOT REMOVE LIKE THIS
        m_coneFrame[i].second.setValidState(false);
        continue; 
      }
      if(!m_coneFrame[i].first && m_coneFrame[i].second.shouldBeInFrame()){  
        //std::cout << "is not matched but should be in frame | curr X: " << m_coneFrame[i].second.getX() << " shift: "  << posShiftX/m<< std::endl;
        double x = m_coneFrame[i].second.getX() - posShiftX/m;
        double y = m_coneFrame[i].second.getY() - posShiftY/m;
        m_coneFrame[i].second.setX(x);
        m_coneFrame[i].second.setY(y);
        sentCounter++;    
        m_sentCones.push_back(m_coneFrame[i].second);         
      }
      else if(m_coneFrame[i].first){
        sentCounter++;
        m_coneFrame[i].first = false;
        m_sentCones.push_back(m_coneFrame[i].second);         
      }
      if(!m_coneFrame[i].first){
        m_coneFrame[i].second.addMiss();
      }
    }
  }//loop
  SendEnvelopes(m_sentCones);
}

void Attention::SendEnvelopes(std::vector<Cone> cones){
  for(uint32_t i = 0; i<cones.size(); i++){
    uint32_t index = cones.size()-1-i;
    double x = cones[index].getX();
    double y = cones[index].getY();
    double z = cones[index].getZ();
    Eigen::Vector3f conePoint = Cartesian2Spherical(x,y,z);
    //std::chrono::system_clock::time_point tp = m_CPCReceivedLastTime;
    //cluon::data::TimeStamp sampleTime = cluon::time::convert(tp);
    opendlv::logic::perception::ObjectDirection coneDirection;
    coneDirection.objectId(index);
    coneDirection.azimuthAngle(-conePoint(1));   //Set Negative to make it inline with coordinate system used
    coneDirection.zenithAngle(conePoint(2));
    m_od4.send(coneDirection,m_CPCReceivedLastTime,m_senderStamp);
    opendlv::logic::perception::ObjectDistance coneDistance;
    coneDistance.objectId(index);
    coneDistance.distance(conePoint(0));
    m_od4.send(coneDistance,m_CPCReceivedLastTime,m_senderStamp);
    std::cout << "Sending cone with ID " << index << std::endl;
  }
}

Eigen::Vector3f Attention::Cartesian2Spherical(double &x, double &y, double &z)
{
  float distance = static_cast<float>(sqrt(x*x+y*y+z*z));
  float azimuthAngle = static_cast<float>(atan2(x,y)*static_cast<double>(RAD2DEG));
  float zenithAngle = static_cast<float>(atan2(z,sqrt(x*x+y*y))*static_cast<double>(RAD2DEG));
  Eigen::Vector3f pointInSpherical;
  pointInSpherical << distance,azimuthAngle,zenithAngle;
  return pointInSpherical;

}

Eigen::MatrixXd Attention::RANSACRemoveGround(Eigen::MatrixXd pointCloudInRANSAC)
{

  Eigen::MatrixXd foundPlane(1,4), planeBest(1,4), planeBestBest(1,4), normal(1,3), pointOnPlane(1,3), indexRangeBest ,indexOutliers(pointCloudInRANSAC.rows(),1);
  foundPlane << 0,0,0,0;
  normal << 0,0,1;
  double d;
  int indexDotFound = 0;
  int planeCounter = 0;
  int M = 1;
  int sizeCloud = pointCloudInRANSAC.rows()-1;
  Eigen::MatrixXd drawnSamples = Eigen::MatrixXd::Zero(3,3);
  Eigen::MatrixXd distance2Plane = Eigen::MatrixXd::Zero(pointCloudInRANSAC.rows(),1);
  Eigen::MatrixXd dotProd = Eigen::MatrixXd::Zero(pointCloudInRANSAC.rows(),1);
  Eigen::MatrixXd indexDot = Eigen::MatrixXd::Zero(pointCloudInRANSAC.rows(),1);

  Eigen::Vector3d planeFromSamples, v0, v1, v2, crossVec1, crossVec2, crossCoefficients;
  int outliersFound, inliersFound;
  double normalBest, normalBestLast;
  normalBestLast = 10000;
  for(int i = 0; i < m_ransacIterations; i++)
  {
    outliersFound = 0;
    inliersFound = 0;
    indexOutliers = Eigen::MatrixXd::Zero(pointCloudInRANSAC.rows(),1);
    for(int j = 0; j < 3; j++){

      int indexShuffle = M + rand() / (RAND_MAX / (sizeCloud - M + 1) + 1);

      drawnSamples.row(j) = pointCloudInRANSAC.row(indexShuffle);
    }
    v0 << drawnSamples(0,0), drawnSamples(0,1), drawnSamples(0,2);
    v1 << drawnSamples(1,0), drawnSamples(1,1), drawnSamples(1,2);
    v2 << drawnSamples(2,0), drawnSamples(2,1), drawnSamples(2,2);
    crossVec1 = v0-v1;
    crossVec2 = v0-v2;
    crossCoefficients = crossVec2.cross(crossVec1);
    crossCoefficients = crossCoefficients.normalized();
    d = v0.dot(crossCoefficients);
    d = d*-1;
    foundPlane << crossCoefficients(0), crossCoefficients(1), crossCoefficients(2), d;
    //Calculate perpendicular distance to found plane
    for(int p = 0; p < pointCloudInRANSAC.rows(); p++){
      distance2Plane(p,0) = fabs(foundPlane(0,0)*pointCloudInRANSAC(p,0) + foundPlane(0,1)*pointCloudInRANSAC(p,1) + foundPlane(0,2)*pointCloudInRANSAC(p,2) + foundPlane(0,3))/crossCoefficients.norm();
      //Find index of inliers
      if(distance2Plane(p,0) >= m_inlierRangeThreshold){
        indexOutliers(outliersFound,0) = p;
        outliersFound++;

      }

    }
    if(outliersFound > 0){
      Eigen::MatrixXd indexRange = Eigen::MatrixXd::Zero(outliersFound,1);
      indexRange = indexOutliers.topRows(outliersFound+1);

      inliersFound = pointCloudInRANSAC.rows()-outliersFound;
      if(inliersFound > m_inlierFoundTreshold ){

        planeBest = foundPlane;
        normalBest = sqrt( (planeBest(0,0)-normal(0,0))*(planeBest(0,0)-normal(0,0)) + (planeBest(0,1)-normal(0,1))*(planeBest(0,1)-normal(0,1)) + (planeBest(0,2)-normal(0,2))*(planeBest(0,2)-normal(0,2)));

        if(normalBest < normalBestLast){
          normalBestLast = normalBest;
          planeBestBest = planeBest;
          indexRangeBest.resize(indexRange.rows(),indexRange.cols());
          indexRangeBest = indexRange;
          pointOnPlane = drawnSamples.row(0);
          planeCounter++;

        }

      }
    }
    else{

      if(planeCounter > 0){

        m_lastBestPlane = planeBestBest;
      }
    }
    //std::cout << "NUmber of iterations is: " << i << std::endl;
  }
  if(planeCounter == 0){

    planeBestBest = m_lastBestPlane;
  }
  for(int p = 0; p < pointCloudInRANSAC.rows(); p++){

    dotProd(p,0) = planeBestBest(0,0)*(pointCloudInRANSAC(p,0)-pointOnPlane(0,0)) + planeBestBest(0,1)*(pointCloudInRANSAC(p,1)-pointOnPlane(0,1)) + planeBestBest(0,2)*(pointCloudInRANSAC(p,2)-pointOnPlane(0,2));

    if(dotProd(p,0) > m_dotThreshold){

      indexDot(indexDotFound,0) = p;
      indexDotFound++;
    }
  }

  Eigen::MatrixXd sortedIndex;
  if(indexDotFound > 0 && indexRangeBest.rows() > 0){
  Eigen::MatrixXd indexDotter = Eigen::MatrixXd::Zero(indexDot.rows(),indexDot.cols());
  indexDotter = indexDot.topRows(indexDotFound+1);
  sortedIndex = Eigen::MatrixXd::Zero(indexDotter.rows()+indexRangeBest.rows(),1); //index2Keep
  sortedIndex << indexRangeBest,  //index2Keep
                 indexDotter;
  }else if(indexDotFound < 0.1 && indexRangeBest.rows() > 0){

    sortedIndex = Eigen::MatrixXd::Zero(indexRangeBest.rows(),1); //index2Keep
    sortedIndex << indexRangeBest;  //index2Keep
                  
  }else if(indexDotFound > 0 && indexRangeBest.rows() < 0.1){
    Eigen::MatrixXd indexDotter = Eigen::MatrixXd::Zero(indexDot.rows(),indexDot.cols());
    indexDotter = indexDot.topRows(indexDotFound+1);

    sortedIndex = Eigen::MatrixXd::Zero(indexDotter.rows(),1); //index2Keep
    sortedIndex << indexDotter;  //index2Keep
  }
  //Remove duplicates
  //Eigen::MatrixXd sortedIndex = RemoveDuplicates(index2Keep);
  //Remove found inlier index from
  Eigen::MatrixXd pcRefit = Eigen::MatrixXd::Zero(sortedIndex.rows(),3); //sortedIndex
  for(int i = 0; i < sortedIndex.rows(); i++){
    pcRefit.row(i) = pointCloudInRANSAC.row(static_cast<int>(sortedIndex(i)));

  }
  std::cout << "Ground plane found:" << std::endl;
  std::cout << planeBestBest << std::endl;

  return pcRefit;

}
Eigen::MatrixXd Attention::layerRemoveGround(Eigen::MatrixXd pointCloudConeROI)
{
  uint32_t startLayer = 6;
  uint32_t endLayer = 10; 

  std::vector<uint32_t> saveAllIndex;
  std::vector<uint32_t> triPointVector;
  


  std::vector<std::vector<uint32_t>> clusters;
  uint32_t clusterCounter = 0;
  std::vector<uint32_t> saveLayerIndex;
  for(uint32_t j = startLayer; j < endLayer; j++){
    
     

    for(uint32_t i = j; i < pointCloudConeROI.rows(); i = i+16){
      //std::cout << "x: " << pointCloudConeROI(i,0) << " y: " << pointCloudConeROI(i,1) << std::endl;
      double distance = std::sqrt(pointCloudConeROI(i,0)*pointCloudConeROI(i,0) + pointCloudConeROI(i,1)*pointCloudConeROI(i,1));
      double azimuthAngle =(atan2(pointCloudConeROI(i,1),pointCloudConeROI(i,0)));
      //std::cout << "Distance: " << distance << " Azimuth: " << azimuthAngle << std::endl;

      if(j == 9 ){ //endlayer
        if(distance > 0.1 && distance < 7 && std::fabs(azimuthAngle) < DEG2RAD*180 ){
          saveLayerIndex.push_back(i);
        }
      }
      if( j == 8 || j == 7 || j == 6 ){
        if(distance > 0.1 && distance < 20 && std::fabs(azimuthAngle) < DEG2RAD*180 ){
        saveLayerIndex.push_back(i);
        }
      }  



    }

     //std::cout << "azimuts: " << m_numberOfAzimuths << std::endl;
    //Eigen::MatrixXd currLayerPoints = Eigen::MatrixXd::Zero(saveLayerIndex.size(),3);
     //std::cout << "SaveIndexSize: " << saveLayerIndex.size() << std::endl; 
     std::vector<uint32_t> localClusterVector;
     clusters.push_back(localClusterVector);
    for(uint32_t i = 1; i < saveLayerIndex.size(); i++){

      double distance = CalculateXYDistance(pointCloudConeROI,saveLayerIndex[i-1] , saveLayerIndex[i]);
       
      if(std::abs(distance) < 0.3){
        localClusterVector.push_back(saveLayerIndex[i-1]);

        //std::cout << "Cluster: " << clusterCounter << std::endl;
      }else{

        localClusterVector.push_back(saveLayerIndex[i-1]);
        clusters[clusterCounter] = localClusterVector;
        localClusterVector.clear();
        clusterCounter++;
        clusters.push_back(localClusterVector);
        //std::cout << "ClusterCounter: " << clusterCounter << std::endl;
      }

    }


  }

    return filterClusters(clusters);
}

Eigen::MatrixXd Attention::filterClusters(std::vector<std::vector<uint32_t>> clusterVector){

    std::vector<uint32_t> notFloorIndex;
    for(uint32_t i = 0; i < clusterVector.size(); i++){
      if(clusterVector[i].size() > 0){
      double clusterMeanDistance = getClusterMeanDist(clusterVector[i]);
      double clusterSize = static_cast<double>(clusterVector[i].size());

      double potentialMaxDist = clusterSize*(clusterMeanDistance*0.2*DEG2RAD);
      if(clusterSize < 45 && potentialMaxDist <= 0.2){

        for(uint32_t j = 0; j < clusterSize; j++){

          notFloorIndex.push_back(clusterVector[i][j]);
        }

      }

    }

  }

  Eigen::MatrixXd filteredClusterEigen = Eigen::MatrixXd::Zero(notFloorIndex.size(),3);

  for(uint32_t i = 0; i < notFloorIndex.size(); i++){

    filteredClusterEigen.row(i) = m_pointCloud.row(notFloorIndex[i]);

  }

  return filteredClusterEigen;
}

double Attention::getClusterMeanDist(std::vector<uint32_t> thisCluster){
  double d = 0;
  uint32_t size = thisCluster.size();
  for(uint32_t i = 0; i < size; i++){

    d = d + std::sqrt( m_pointCloud(thisCluster[i],0)*m_pointCloud(thisCluster[i],0) + m_pointCloud(thisCluster[i],1)*m_pointCloud(thisCluster[i],1) );


  }

  return d/size;
}
Eigen::MatrixXd Attention::RemoveDuplicates(Eigen::MatrixXd needSorting)
{

  std::vector<double> vect;

  for(int i=0; i< needSorting.rows(); i++){

    vect.push_back(needSorting(i,0));

  }

  sort(vect.begin(),vect.end());
  vect.erase(unique(vect.begin(),vect.end()),vect.end());
  needSorting.resize(vect.size(),1);

  for(unsigned int i=0; i< vect.size(); i++){

    needSorting(i,0)=vect.at(i);

  }

  return needSorting;

}

Eigen::MatrixXd Attention::getFullPointCloud(){
  std::lock_guard<std::mutex> lock(m_cpcMutex);
  return m_pointCloud;
}


Eigen::MatrixXd Attention::getROIPointCloud(){
  std::lock_guard<std::mutex> lock(m_drawerMutex);
  return m_pointCloudROI;
}

Eigen::MatrixXd Attention::getRANSACPointCloud(){
  std::lock_guard<std::mutex> lock(m_drawerMutex);
  return m_pcAfterRANSAC;
}

std::vector<Cone> Attention::getSentCones(){
  std::lock_guard<std::mutex> lock(m_drawerMutex);
  return m_sentCones;
}