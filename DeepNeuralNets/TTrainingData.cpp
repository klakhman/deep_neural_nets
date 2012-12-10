#include "TTrainingData.h"
#include <string>
#include <cstdlib>

using namespace std;

// Загрузка обучающей выборки
void TTrainingData::loadTrainingSet(istream& inputSource){
  string tmpStr;
  inputSource >> tmpStr;
  trainingDataSize = atoi(tmpStr.c_str());
  inputSource >> tmpStr;
  inputResolution = atoi(tmpStr.c_str());
  inputSource >> tmpStr;
  outputResolution = atoi(tmpStr.c_str());
  dataSet.resize(trainingDataSize);
  for (int currentExample = 0; currentExample < trainingDataSize; ++currentExample){
    dataSet[currentExample].input.resize(inputResolution);
    dataSet[currentExample].input.resize(outputResolution);
    for (int currentInput = 0; currentInput < inputResolution; ++currentInput){
      inputSource >> tmpStr;
      dataSet[currentExample].input[currentInput] = atof(tmpStr.c_str());
    }
    for (int currentOutput = 0; currentOutput < outputResolution; ++currentOutput){
      inputSource >> tmpStr;
      dataSet[currentExample].output[currentOutput] = atof(tmpStr.c_str());
    }
  }
}
