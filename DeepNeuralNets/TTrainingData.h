#ifndef T_TRAINING_DATA_H
#define T_TRAINING_DATA_H

#include <vector>
#include <iostream>

// Класс обучающих/тестовых выброк
class TTrainingData{
public:
  // Структура примера обучающей выборки
  struct STrainingExample{
    std::vector<double> input;
    std::vector<double> output;
  };

private:
  // Обучающее множество
  std::vector<STrainingExample> dataSet;
  // Размер обучающего множества
  int trainingDataSize;
  
  int inputResolution;
  int outputResolution;

public:
  TTrainingData(){
    trainingDataSize = 0;
    inputResolution = 0;
    outputResolution = 0;
  }
  ~TTrainingData(){}

  // Загрузка обучающей выборки
  void loadTrainingSet(std::istream& inputSource);

};

#endif // T_TRAINNING_DATA_H