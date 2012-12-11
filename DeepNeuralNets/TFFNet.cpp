#include "TFFNet.h"
#include "TTrainingData.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "armadillo"

using namespace std;

// TODO (11.12.12): 
// 1) Заменить везде arma::Mat::resize() на ::set_size (работает быстрее)
// 2) Заменить везде где возможно (размер матрицы изначально установлен и известен) заменить простое присваинвание на присваивание 
//    с указанием диапазона левого операнда (submat() или subvec()) - это скорее всего должно работать быстрее

// Загрузка сети из файла
void TFFNet::loadNetwork(istream& inputSource){
  string tmpStr;
  inputSource >> tmpStr;
  layersQuantity = atoi(tmpStr.c_str());
  weightsMatrices.resize(layersQuantity);
  neuronsDistribution.resize(layersQuantity + 1);
  // Сначала заполняем распределение нейронов по слоям
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer){
    inputSource >> tmpStr;
    neuronsDistribution[currentLayer] = atoi(tmpStr.c_str());
  }
  // Заполняем матрицы весов
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    weightsMatrices[currentLayer].set_size(neuronsDistribution[currentLayer + 1], neuronsDistribution[currentLayer] + 1);
    for (int currentPostNeuron = 1; currentPostNeuron <= neuronsDistribution[currentLayer + 1]; ++currentPostNeuron)
      for (int currentPreNeuron = 0; currentPreNeuron <= neuronsDistribution[currentLayer]; ++currentPreNeuron){
        inputSource >> tmpStr;
        weightsMatrices[currentLayer](currentPostNeuron - 1, currentPreNeuron) = atof(tmpStr.c_str());
      }
  }
  // Создаем структуру выходов нейронов
  neuronsOutputs.resize(layersQuantity + 1);
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer){
    neuronsOutputs[currentLayer].set_size(neuronsDistribution[currentLayer]);
    neuronsOutputs[currentLayer].fill(0);
  }
}

// Выгрузка сети в файл
void TFFNet::printNetwork(std::ostream& outputSource){
  outputSource << layersQuantity << endl;
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer)
    outputSource << neuronsDistribution[currentLayer] << "\t";
  outputSource << endl;
  // Записываем матрицы весов
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    for (int currentPostNeuron = 1; currentPostNeuron <= neuronsDistribution[currentLayer + 1]; ++currentPostNeuron){
      for (int currentPreNeuron = 0; currentPreNeuron <= neuronsDistribution[currentLayer]; ++currentPreNeuron)
        outputSource << weightsMatrices[currentLayer](currentPostNeuron - 1, currentPreNeuron)  << "\t";
      outputSource << endl;
    }
    outputSource << endl;
  }
}

// Генерация сети со случайными весами
// layersQuantity - кол-во слоев, не считая входного; weightsInitValue - дипапазон рандомизации весов связей (-weightsInitValue; weightsInitValue)
void TFFNet::generateRandomNet(int _layersQuantity, const std::vector<int>& _neuronsDistribution, double weightsInitValue /*=0.1*/){
  layersQuantity = _layersQuantity;
  weightsMatrices.resize(layersQuantity);
  neuronsDistribution = _neuronsDistribution;

  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    weightsMatrices[currentLayer] = arma::randu< arma::Mat<double> >(neuronsDistribution[currentLayer + 1], neuronsDistribution[currentLayer] + 1);
    // Armagillo генерирует случаные матрицы в диапазоне [0, 1] - приводим к нужному нам диапазону
    weightsMatrices[currentLayer] = weightsMatrices[currentLayer] * (2 * weightsInitValue) -  weightsInitValue; 
  }
  // Создаем структуру выходов нейронов
  neuronsOutputs.resize(layersQuantity + 1);
  for (int currentLayer = 0; currentLayer <= layersQuantity; ++currentLayer){
    neuronsOutputs[currentLayer].set_size(neuronsDistribution[currentLayer]);
    neuronsOutputs[currentLayer].fill(0);
  }
}

// Обсчет сети (возвращает выход сети)
vector<double> TFFNet::calculate(const vector<double>& inputVector){
  // Заполняем выходы нулевого слоя и подготавливаем нулевой вектор
  neuronsOutputs[0].subvec(0, neuronsDistribution[0] - 1)  = arma::conv_to< arma::Col<double> >::from(inputVector);
  // Вектор со входом bias
  arma::Col<double> inputToLayer;

  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    // Заполняем вход для следующего слоя
    inputToLayer.set_size(neuronsDistribution[currentLayer] + 1);
    inputToLayer(0) = 1;
    inputToLayer.subvec(1, neuronsDistribution[currentLayer]) = neuronsOutputs[currentLayer];
    neuronsOutputs[currentLayer + 1].subvec(0, neuronsDistribution[currentLayer + 1] - 1) = logisticFunc(weightsMatrices[currentLayer] * inputToLayer); 
  }
  return arma::conv_to< vector<double> >::from(neuronsOutputs[layersQuantity]);
}

// Подсчет локальных градиентов для всех нейронов сети (при алгоритме обратного распространения)
vector< arma::Col<double> > TFFNet::calculateLocalGradients(const vector<double>& desiredOutput) const{
  vector< arma::Col<double> > localGradients(layersQuantity);

  // Вычисляем градиент для входного слоя
  localGradients[layersQuantity - 1] = 
    (arma::conv_to< arma::Col<double> >::from(desiredOutput) - neuronsOutputs[layersQuantity]) * derLogisticFunc(neuronsOutputs[layersQuantity]);

  // Теперь вычисляем последовательно для всех слоев
  for (int currentLayer = layersQuantity - 2; currentLayer >= 0; --currentLayer)
    // Получаем локальные градиенты транспонируя матрицу весов (все кроме весов от bias) и перемножая с локальными градиентами следующего слоя 
    // Потом каждый элемент перемножаем с производной по выходу
    localGradients[currentLayer] = 
      (trans(weightsMatrices[currentLayer + 1].submat(0, 1, neuronsDistribution[currentLayer + 2] - 1, neuronsDistribution[currentLayer + 1])) * 
      localGradients[currentLayer + 1]) % derLogisticFunc(neuronsOutputs[currentLayer + 1]);

  return localGradients;
}

// Подсчет модификации весов сети на одном примере - передается "матрица" локальных градиентов нейронов сети, возвращает ошибку на примере
double TFFNet::getWeightsDelta_SingleSample(vector< arma::Mat<double> >& weightsDelta, 
                                            const vector<double>& input, const vector<double>& desiredOutput, double learningRate /*=0.1*/){
  vector<double> output = calculate(input);
  vector< arma::Col<double> > localGradients = calculateLocalGradients(desiredOutput);
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    arma::Row<double> outputs(neuronsDistribution[currentLayer] + 1);
    outputs(0) = 1;
    outputs.subvec(1, neuronsDistribution[currentLayer]) = neuronsOutputs[currentLayer];
    weightsDelta[currentLayer] = learningRate * (localGradients[currentLayer] * outputs);
  }
  double squareError = 0;
  for (int currentOutput = 0; currentOutput < neuronsDistribution[layersQuantity]; ++currentOutput)
    squareError += (desiredOutput[currentOutput] - output[currentOutput]) * (desiredOutput[currentOutput] - output[currentOutput]) / 2;
  return squareError;
}

// Подсчет модификации весов сети в пакетном режиме на все обучающем мн-ве - возвращает значение ошибки до обучения
double TFFNet::getWeightsDelta_BatchSet(vector< arma::Mat<double> >& weightsDelta, const TTrainingData& trainingSet, double learningRate /*=0.1*/){
  double squareError = 0;
  for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer){
    weightsDelta[currentLayer].set_size(neuronsDistribution[currentLayer + 1], neuronsDistribution[currentLayer] + 1);
    weightsDelta[currentLayer].fill(0);
  }
  vector< arma::Mat<double> > singleSampleDelta(layersQuantity);
  for (int currentSample = 0; currentSample < trainingSet.getTrainingDataSize(); ++currentSample){
    squareError += getWeightsDelta_SingleSample(singleSampleDelta, trainingSet.getTrainingExample(currentSample + 1).input,
                   trainingSet.getTrainingExample(currentSample + 1).output, learningRate) / trainingSet.getTrainingDataSize();
    for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer)
      weightsDelta[currentLayer] += singleSampleDelta[currentLayer] / trainingSet.getTrainingDataSize();
  }
  return squareError;
}

// Обучение сети классическим методом обратного распространения ошибки
void TFFNet::trainBackProp(const TTrainingData& trainingSet, int trainingEpochsQuantity, double learningRate /*=0.1*/){
  vector< arma::Mat<double> > weightsDelta(layersQuantity);
   
  for (int trainingEpoch = 0; trainingEpoch < trainingEpochsQuantity; ++trainingEpoch){
    double squareError = getWeightsDelta_BatchSet(weightsDelta, trainingSet);
    for (int currentLayer = 0; currentLayer < layersQuantity; ++currentLayer)
      weightsMatrices[currentLayer] += weightsDelta[currentLayer];
    cout << trainingEpoch + 1 << "\t" << squareError << endl;
  }
}

