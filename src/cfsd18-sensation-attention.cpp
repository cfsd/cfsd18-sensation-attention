/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"
#include "cone.hpp"
#include "attention.hpp"

#include <cstdint>
#include <tuple>
#include <utility>
#include <iostream>
#include <string>
#include <thread>

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{0};
  std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if (commandlineArguments.count("cid")<1) {
    std::cerr << argv[0] << " is a lidar cone detection module for the CFSD18 project." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> [--id=<Identifier in case of simulated units>] [--verbose] [Module specific parameters....]" << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --id=120"  <<  std::endl;
    retCode = 1;
  } else {
    bool const VERBOSE{commandlineArguments.count("verbose") != 0};
    (void)VERBOSE;
    // Interface to a running OpenDaVINCI session (ignoring any incoming Envelopes).
    cluon::data::Envelope data;
    //std::shared_ptr<Slam> slammer = std::shared_ptr<Slam>(new Slam(10));
    cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};
    uint32_t attentionStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["id"]));
    Attention attention(commandlineArguments,od4);
    int pointCloudMessages = 0;
    bool readyState = false;
    auto envelopeRecieved{[&senser = attention, &ready = readyState, &counter = pointCloudMessages](cluon::data::Envelope &&envelope)
      {
        senser.nextContainer(envelope);
        if(!ready){
          if(counter > 20){
            ready = true;
            senser.setReadyState(ready);  
            std::cout << "Attention Ready .." << std::endl;        
          }else{counter++;}
        }    
      } 
    };
    od4.dataTrigger(opendlv::proxy::PointCloudReading::ID(),envelopeRecieved);

    // Just sleep as this microservice is data driven.
    using namespace std::literals::chrono_literals;
    
    while (od4.isRunning()) {

      if(readyState){
        opendlv::system::SignalStatusMessage ssm;
        ssm.code(1);
        cluon::data::TimeStamp sampleTime = cluon::time::now();
        od4.send(ssm, sampleTime ,attentionStamp);
      }
      std::this_thread::sleep_for(0.1s);
      std::chrono::system_clock::time_point tp;
    }
  }
  return retCode;
}


