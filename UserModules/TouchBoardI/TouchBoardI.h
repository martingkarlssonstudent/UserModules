//
//  TouchBoardI.h

#ifndef TouchBoardI_
#define TouchBoardI_

#include "IKAROS.h"


class TouchBoardI: public Module
{
public:
    static Module *Create(Parameter * p) { return new TouchBoardI(p); }

    TouchBoardI(Parameter * p) : Module(p) {}
    virtual ~TouchBoardI();

    void Init();
    void Tick();

    Serial *SerialPort;

    char * rcvmsg;

    float *	touchOutput_array;

    int noTicks;
    std::string touchStr;

};

#endif
