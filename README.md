Approximate matrix completion based on the cavity method
===
This repository includes cavity-based matrix factorization (CBMF) and approximate cavity-based matrix factorization (ACBMF) for matrix completion.

## Requirement
- Installing [Almadillo](http://arma.sourceforge.net/), which is c++ library for linear algebra, is needed to run.
- Compiling and running was tested under gcc version 4.2.1 (macOS High Sierra).

## Usage
### Compiling:
Simply use `make`.
### Options:
- -l: Learning rate. [default: 0.3]
- -L: Reguralization parameter. [default: 3]
- -R: Rank. [default: 10]
- -m: Maximum number of iterations. [default: 100]
- -r: Filename of dataset for training. [default: dataset/ml_1m_train.txt]
- -t: Filename of dataset for test. [default: dataset/ml_1m_test.txt]
- -o: Where output files are to be placed. [default: output/]
- -c: Exponent of convergence condition. If RMSE < pow(10, convint) is satisfied, it is regarded as convergence. [default: -5]
- -p: If this option is set, ACBMF is to be performed. Without this option, CBMF is to be done.
- -h: Show help.
### Examples:
Perfoming CBMF using 'dataset/ml_1m_train.txt' as training dataset and 'dataset/ml_1m_test.txt' as test dataset.  
```
./cbmf -r dataset/ml_1m_train.txt -t dataset/ml_1m_test.txt
```
Perfoming ACBMF using 'dataset/ml_1m_train.txt' as training dataset and 'dataset/ml_1m_test.txt' as test dataset.  
```
./cbmf -p -r dataset/ml_1m_train.txt -t dataset/ml_1m_test.txt
```
Showing help.
```
./cbmf -h
```
## Reference
