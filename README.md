## Разработка алгоритма построения карты с точным расположением объектов в векторном формате для автономных машин

### Инструкция по запуску
1. Скачать репозиторий
2. Установить зависимости
```bash
conda env create -f environment.yml
conda activate env0
```
3. Отдельно установить [Segment Anything](https://github.com/facebookresearch/segment-anything)
4. Скачать репозиторий [kitti360scripts](https://github.com/autonomousvision/kitti360Scripts) - для работы с датасетом kitti-360

### Подготовка данных
1. Скачать датасет [kitti-360](http://www.cvlibs.net/datasets/kitti-360/), а именно:
    - `Fisheye Images`
    - `Calibrations`
    - `Vehicle Poses`
    - `Accumulated Point Clouds for Train & Val`
    - `Test Semantic`
2. Запустить скрипт для конвертации парных fisheye в равнопромежуточные проекции 
(предполагается, что изображения находятся в папках `KITTI-360/data_2d_raw/2013_05_28_drive_????_sync/image_0(2|3)/data_rgb/`)

