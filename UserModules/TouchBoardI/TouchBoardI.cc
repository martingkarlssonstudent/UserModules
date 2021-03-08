//
//  TouchBoardContIn.cpp
//
//
//

#include "TouchBoardI.h"


//#include "../../Kernel/IKAROS_Serial.h"

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <array>


using namespace ikaros;
using namespace std;

TouchBoardI::~TouchBoardI()
{
    SerialPort->Close();
}

void TouchBoardI::Init()
{
    SerialPort = new Serial(GetValue("port"), 57600);
    SerialPort->Flush();
    rcvmsg = new char[256];
    touchOutput_array = GetOutputArray("TOUCH_OUTPUT");
}

void TouchBoardI::Tick()
{
  noTicks++;
  if(!SerialPort)
    return; // Not connected to board

  int rate = 256;
  int timeMs = 35;
  //Receives sended char from serial
  int count = SerialPort->ReceiveBytes(rcvmsg, rate, timeMs);

  std::stringstream stream;
  std::string numbers = std::string();
  int noTouchRecords;
  int val;
  int i=0;

  stream << rcvmsg;
  stream >> numbers;

  int noElectrodes = 12;
  int iElectrode = 0;
  std::size_t noCharStart = 0;
  std::size_t noCharEnd = 0;
  std::string delimiter (",");
  std::string evalue;

  bool numbersok = false;
  if (numbers.length())
  {
    noCharStart = numbers.find(delimiter, noCharStart);
    if (noCharStart != 0)
    {
      numbersok = true;
    }
  }

  if (numbersok)
  {
    while(iElectrode < noElectrodes)
    {
      iElectrode++;
      if (iElectrode == 1)
      {
        noCharStart = 0;
        noCharEnd = 0;
      }
      else
      {
        noCharStart = noCharEnd+2;
      }
      if (noCharStart >= numbers.length())
      {
        break;
      }
      noCharEnd = numbers.find(delimiter, noCharStart)-1;
      if (noCharEnd > numbers.length())
      {
        noCharEnd = numbers.length()-1;
      }

      int charlength = noCharEnd-noCharStart;
      evalue = numbers.substr(noCharStart, charlength+1);

      touchOutput_array[iElectrode] = std::stof(evalue);
    }
  }
}

static InitClass init("TouchBoardI", &TouchBoardI::Create, "Source/UserModules/TouchBoardI/");
