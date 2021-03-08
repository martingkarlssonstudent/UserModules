#ifndef TouchId_H
#define TouchId_H

#include "IKAROS.h"
#include <stdio.h>

class TouchId: public Module
{
public:
    TouchId(Parameter * p);
    virtual ~TouchId();
    static Module * Create(Parameter * p);

    void 		Init();
    void 		Tick();

    float *     touchInput_array;
    int         touchInput_NoCols;

    float **    touchLong_matrix;
    int         touchLong_NoCols;

    float **    touchCertainty_matrix;
    int         touchCertainty_NoRows;

    float **    touchEmotion_matrix;
    int         noTouches;
    int         noEmotions;

    void        CheckParameters(void);

    int touchType;
    int noTouchTypes;
    int noTouchTypesOut;
    bool touchActive;
    float * touchFinished_array;

private:
    int noTicks;
    int touchLong_matrix_row; // The row count, determining which row is going to be filled in touchLong_matrix
};

#endif
