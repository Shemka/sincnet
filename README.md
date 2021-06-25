# SincNet Experiments
В этом репозитории находится решение тестового задания по статье ["Speaker Recognition From Raw Waveform With SincNet"](https://arxiv.org/pdf/1808.00158.pdf). 
Все реализовано на pytorch + pytorch_lightning + torchaudio.

**train.ipynb** — основной файл содержащий весь процессинг и обучение моделей. Перед его запуском убедитесь, что распаковали датасет LibriSpeech в корневую директорию.

**Report.pdf** — отчет по экспериментам

**model.py** — файл, содержащий код самой модели

**sincnet.py** — файл, содержащий реализацию SincConv авторами статьи (ссылка на авторов в коде)
