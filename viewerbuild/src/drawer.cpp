#include "drawer.hpp"


Drawer::Drawer(std::map<std::string,std::string> commandlineArgs){
    std::cout << commandlineArgs.count("cid") << std::endl;
}


void Drawer::drawRawPoints(){
    std::lock_guard<std::mutex> lock(m_rawMutex);
    uint32_t nPoints = static_cast<unsigned int>(m_rawPoints.rows());
    if(nPoints == 0){
        return;
    }    
    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    for(uint32_t i = 0; i<nPoints; i++){
        float x = static_cast<float>(m_rawPoints(i,0));
        float y = static_cast<float>(m_rawPoints(i,1));
        float z = static_cast<float>(m_rawPoints(i,2));
        glVertex3f(x,y,z);
    }
    glEnd();
}

void Drawer::drawROIPoints(){
    std::lock_guard<std::mutex> lock(m_ROIMutex);
    uint32_t nPoints = static_cast<unsigned int>(m_ROIPoints.rows());
    if(nPoints == 0){
        return;
    }    
    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(0.5,0.5,0.0);
    for(uint32_t i = 0; i<nPoints; i++){
        float x = static_cast<float>(m_ROIPoints(i,0));
        float y = static_cast<float>(m_ROIPoints(i,1));
        float z = static_cast<float>(m_ROIPoints(i,2));
        glVertex3f(x,y,z);
    }
    glEnd();
}


void Drawer::drawRANSACPoints(){
    std::lock_guard<std::mutex> lock(m_RANSACMutex);
    uint32_t nPoints = static_cast<unsigned int>(m_RANSACPoints.rows());
    if(nPoints == 0){
        return;
    }    
    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(0.5,0.5,0.0);
    for(uint32_t i = 0; i<nPoints; i++){
        float x = static_cast<float>(m_RANSACPoints(i,0));
        float y = static_cast<float>(m_RANSACPoints(i,1));
        float z = static_cast<float>(m_RANSACPoints(i,2));
        glVertex3f(x,y,z);
    }
    glEnd();
}

void Drawer::drawCones(){
    std::lock_guard<std::mutex> lock(m_coneMutex);
    uint32_t nPoints = m_cones.size();
    if(nPoints == 0){
        return;
    }    
    glPointSize(10);
    glBegin(GL_POINTS);
    glColor3f(0.5,0.5,1);
    for(uint32_t i = 0; i<nPoints; i++){
        float x = static_cast<float>(m_cones[i].getX());
        float y = static_cast<float>(m_cones[i].getY());
        float z = static_cast<float>(m_cones[i].getZ());
        glVertex3f(x,y,z);
    }
    glEnd();
}

void Drawer::setRawPoints(Eigen::MatrixXd rawPoints){
    std::lock_guard<std::mutex> lock(m_rawMutex);
    m_rawPoints = rawPoints;
}


void Drawer::setROIPoints(Eigen::MatrixXd ROIPoints){
    std::lock_guard<std::mutex> lock(m_ROIMutex);
    m_ROIPoints = ROIPoints;
}


void Drawer::setRANSACPoints(Eigen::MatrixXd RANSACPoints){
    std::lock_guard<std::mutex> lock(m_RANSACMutex);
    m_RANSACPoints = RANSACPoints;
}


void Drawer::setCones(std::vector<Cone> cones){
    std::lock_guard<std::mutex> lock(m_coneMutex);
    m_cones.clear();
    for(uint32_t i = 0; i<cones.size(); i++)
        m_cones.push_back(cones[i]);
}