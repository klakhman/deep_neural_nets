#ifndef TFFNET_H
#define TFFNET_H

#include <iostream>
#include <vector>
#include "armadillo"

// Класс нейронной сети прямого распространения
class TFFNet{
  // Матрицы весов связей между слоями
  // Кол-во матриц layersQuantity. Первая матрица - это веса от входов к первому слою.
  // Веса для одного нейрона расположены в строках матриц, где нулевой элемент строки - это bias.
  std::vector< arma::Mat<double> > weightsMatrices;
  // Распределение нейронов по слоям - layersQuantity + 1, не включая bias (нулевой элемент указывает на кол-во входов в сеть)
  std::vector<int> neuronsDistribution;
  // Кол-во слоев в сети (не считая входной)
  int layersQuantity;
  // Текущие выходы нейронов во всех слоях
  std::vector< std::vector<double> > neuronsOutputs;

  // Функция выхода логистической функции (на векторе)
  arma::Col<double> logisticFunc(const arma::Col<double>& inputCol, int slope = 2){
    return 1 / (exp(inputCol * -slope) + 1);
  }

public:
  TFFNet(){
    layersQuantity = 0;
  }
  ~TFFNet(){}
  
  // Возвращает кол-во слоев (не учитывая входной)
  int getLayersQuantity() const{
    return layersQuantity;
  }

  //Возвращает кол-во нейронов в слое (нулевой слой - входной)
  int getNeuronsQuantity(int layer) const{
    if ((layer > layersQuantity) || (!layersQuantity)) {
      std::cout << "Error: Trying to access layer beyond the existing..." << std::endl;
      exit(1);
    }
    return neuronsDistribution[layer];
  }

  // Загрузка сети из файла
  void loadNetwork(std::istream& inputSource);
  // Выгрузка сети в файл
  void printNetwork(std::ostream& outputSource);
  // Обсчет сети (возвращает выход сети)
  std::vector<double> calculate(const std::vector<double>& inputVector);

  // Возвращает текущий выходной вектор (полученный после последнего обсчета)
  std::vector<double> getOutputVector() const{
    if (!layersQuantity){
      std::cout << "Error: Trying to access layer beyond the existing..." << std::endl;
      exit(1);
    }
    return neuronsOutputs[layersQuantity];
  }

};

#endif // TFFNET_H