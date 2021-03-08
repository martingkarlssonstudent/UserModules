//
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

// using namespace std;
using namespace tensorflow;
using namespace std;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;

#include "TouchAnalysis.h"
#include <math.h>
#include <sstream>
#include <string>
#include <list>

using namespace ikaros;

Module *
TouchAnalysis::Create(Parameter * p)
{
	return new TouchAnalysis(p);
}

TouchAnalysis::TouchAnalysis(Parameter * p): Module(p)
{
    CheckParameters();
}

void TouchAnalysis::Init()
{
//	touchInput_array = GetInputArray("TOUCH_INPUT");
//	touchInput_NoCols = GetInputSize("TOUCH_INPUT");

	touchActive = GetInputArray("TOUCHACTIVE_INPUT");

	touchCertainty_matrix = GetInputMatrix("TOUCHCERTAINTY_INPUT");
	touchCertainty_NoRows = GetInputSizeX("TOUCHCERTAINTY_INPUT");

	touchEmotion_matrix = GetInputMatrix("TOUCHEMOTION_INPUT");
	noTouches = GetInputSizeX("TOUCHEMOTION_INPUT");
	noEmotions = GetInputSizeY("TOUCHEMOTION_INPUT");

	emotionRGB_matrix = GetInputMatrix("EMOTIONRGB_INPUT");
	noRGB = GetInputSizeY("EMOTIONRGB_INPUT");

	emotionPupilSize_array = GetInputArray("EMOTIONPUPILSIZE_INPUT");

	epi_eye_r_array = GetOutputArray("EPI_EYE_R");
	epi_eye_g_array = GetOutputArray("EPI_EYE_G");
	epi_eye_b_array = GetOutputArray("EPI_EYE_B");

	epi_pupilsize_array = GetOutputArray("EPI_PUPILSIZE");

	Bind(noTouchTypes, "noTouchTypes");
}

TouchAnalysis::~TouchAnalysis()
{
	// Destroy data structures that is allocated in Init.
//	destroy_matrix(touchLong_matrix);
}

void TouchAnalysis::Tick()
{
//	int touchSignal = 0;
//	bool touchActive = false;

//	cout << noTicks;
//	std::cout << '\n';
	if (touchActive[0] > 0)
	{
		noTouchTypesOut =	noTouchTypes;

/*	if (noTicks > 0)
	{
		touchActive = true;
	}
	else
	{
		for (int i=0; i<touchInput_NoCols; i++)
		{
			touchSignal = touchInput_array[i];
			if (touchSignal > 0)
			{
				touchActive = true;
				cout << "Touch active";
				std::cout << '\n';
				break;
			}
		}
	}
*/
//	if (touchActive)
//	{
/*		if (noTicks == 0)
		{
			touchLong_NoCols = 100;
			touchLong_matrix = create_matrix(touchInput_NoCols, touchLong_NoCols); //Creates a new internal matrix
		}
*/
		noTicks++;
/*	  for (int i=0; i<touchInput_NoCols; i++)
		{
	      touchLong_matrix[touchLong_matrix_row][i] = touchInput_array[i];
		} */

//		touchLong_matrix_row++;
//		if (touchLong_matrix_row == touchLong_NoCols)
//		{
			cout << "noTouchTypes: " << noTouchTypes;
			std::cout << '\n';
			noTicks = 0;

/*			char const *testchar = "test";
			char* argvdefault[] = {strdup("abc")};

			for(int i = 0; i < touchCertainty_NoRows; i++)
			{
				touchCertainty_matrix[0][i] = 0; //Resets first row of 'active' electrodes in touchCertainty_matrix
			}

			rungraph(1, argvdefault, touchLong_matrix, touchInput_NoCols, touchLong_matrix_row, touchCertainty_matrix, noTouchTypesOut);
*/
			float maxCertainty = 0;
			cout << "touchCertainty matrix";
			std::cout << '\n';
			cout << "touchCertainty_NoRows: " << touchCertainty_NoRows;
			std::cout << '\n';
			for (int i=0; i<touchCertainty_NoRows; i++)
			{
					cout << touchCertainty_matrix[0][i] << " ";
					if (touchCertainty_matrix[0][i] > maxCertainty)
					{
						maxCertainty = touchCertainty_matrix[0][i];
						touchType = i;
					}
			}
			std::cout << '\n';

			std::cout << '\n';
			cout << "touchEmotion_matrix"<< '\n';
			for (int i=0; i<noEmotions; i++)
			{
				for (int j=0; j<noTouches; j++)
				{
					cout << touchEmotion_matrix[i][j] << " ";
				}
				std::cout << '\n';
			}
			std::cout << '\n';

			float ** touchEmotion_matrix_T = transpose(create_matrix(noEmotions, noTouches), touchEmotion_matrix, noEmotions, noTouches);

			cout << "touchEmotion_matrix_T" << '\n';
			for (int i=0; i<noTouches; i++)
			{
				for (int j=0; j<noEmotions; j++)
				{
					cout << touchEmotion_matrix_T[i][j] << " ";
				}
				std::cout << '\n';
			}
			std::cout << '\n';

			float ** certaintyEmotion_matrix = multiply(create_matrix(noEmotions, 1), touchCertainty_matrix, touchEmotion_matrix_T, noEmotions, 1, noTouches);

			cout << "certaintyEmotion_matrix"<< '\n';
			for (int i=0; i<noEmotions; i++)
			{
					cout << certaintyEmotion_matrix[0][i] << " ";
			}
		  std::cout << '\n';

			float certaintyEmotionScale = 0;
			for (int i=0; i<noEmotions; i++)
			{
					certaintyEmotionScale += certaintyEmotion_matrix[0][i];
			}

			cout << "certaintyEmotionScale "<< certaintyEmotionScale << '\n';

		  std::cout << '\n';
			cout << "emotionRGB_matrix"<< '\n';
			for (int i=0; i<noRGB; i++)
			{
				for (int j=0; j<noEmotions; j++)
				{
					cout << emotionRGB_matrix[i][j] << " ";
				}
				std::cout << '\n';
			}
		  std::cout << '\n';

			float ** emotionRGB_matrix_T = transpose(create_matrix(noRGB, noEmotions), emotionRGB_matrix, noRGB, noEmotions);

			cout << "emotionRGB_matrix_T"<< '\n';
			for (int i=0; i<noEmotions; i++)
			{
				for (int j=0; j<noRGB; j++)
				{
					cout << emotionRGB_matrix_T[i][j] << " ";
				}
				std::cout << '\n';
			}
		  std::cout << '\n';

// multiply(output, input1, input2, size2_x, size1_y, size1_x);
			float ** certaintyRGB_matrix = multiply(create_matrix(noRGB, 1), certaintyEmotion_matrix, emotionRGB_matrix_T, noRGB, 1, noEmotions);

			cout << "certaintyRGB_matrix/certaintyEmotionScale"<< '\n';
			for (int j=0; j<noRGB; j++)
			{
				cout << certaintyRGB_matrix[0][j]/certaintyEmotionScale << " ";
			}
			std::cout << '\n';

			epi_eye_r_array[0] = certaintyRGB_matrix[0][0]/certaintyEmotionScale;
			epi_eye_b_array[0] = certaintyRGB_matrix[0][1]/certaintyEmotionScale;
			epi_eye_g_array[0] = certaintyRGB_matrix[0][2]/certaintyEmotionScale;

//			destroy_matrix(touchEmotion_matrix);

			std::cout << '\n';
			cout << "emotionPupilSize_array"<< '\n';
			for (int i=0; i<noEmotions; i++)
			{
				cout << emotionPupilSize_array[i] << " ";
			}
			std::cout << '\n';

			float * certaintyPupilsize_array = multiply(create_array(1), certaintyEmotion_matrix, emotionPupilSize_array, noEmotions, 1);

			std::cout << '\n';
			cout << "certaintyPupilsize_array"<< '\n';
			cout << certaintyPupilsize_array[0]<< '\n';

			epi_pupilsize_array[0] = certaintyPupilsize_array[0]/20;

//			touchLong_matrix_row = 0;
//			destroy_matrix(touchLong_matrix);
			destroy_matrix(touchEmotion_matrix_T);
			destroy_matrix(certaintyEmotion_matrix);
			destroy_matrix(emotionRGB_matrix_T);
			destroy_matrix(certaintyRGB_matrix);
			destroy_array(certaintyPupilsize_array);
//		}
//	}
	}
}

void TouchAnalysis::CheckParameters()
{
    touchType = GetIntValueFromList("touchType", "0/1/2/3/4/5/6/7/8/9/10");
		cout << "Touch type parameter:" << touchType;
		std::cout << '\n';
}

static InitClass init("TouchAnalysis", &TouchAnalysis::Create, "Source/UserModules/TouchAnalysis/");
