#ifndef TouchAnalysis_H
#define TouchAnalysis_H

#include "IKAROS.h"
#include <stdio.h>

class TouchAnalysis: public Module
{
public:
    TouchAnalysis(Parameter * p);
    virtual ~TouchAnalysis();
    static Module * Create(Parameter * p);

    void 		Init();
    void 		Tick();
/*
    float *     touchInput_array;
    int         touchInput_NoCols;

    float **    touchLong_matrix;
    int         touchLong_NoCols; */

    float *     touchActive;

    float **    touchCertainty_matrix;
    int         touchCertainty_NoRows;

    float **    touchEmotion_matrix;
    int         noTouches;
    int         noEmotions;

    float **    emotionRGB_matrix;
    int         noRGB;

    float *     emotionPupilSize_array;

    float *     epi_eye_r_array;
    float *     epi_eye_g_array;
    float *     epi_eye_b_array;

    float *     epi_pupilsize_array;

    void        CheckParameters(void);

    int touchType;
    int noTouchTypes;
    int noTouchTypesOut;

private:
    int noTicks;
    int touchLong_matrix_row; // The row count, determining which row is going to be filled in touchLong_matrix
};

#endif
