## Разработка алгоритма построения карты с точным расположением объектов в векторном формате для автономных машин

### Инструкция по запуску
1. Скачать репозиторий и перейдите в него
2. Установить зависимости (установите [python3.8](https://www.python.org/downloads/release/python-380/))
```bash
# виртуальное окружение
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Отдельно установить [Segment Anything](https://github.com/facebookresearch/segment-anything)
4. Скачать репозиторий [kitti360scripts](https://github.com/autonomousvision/kitti360Scripts) - для работы с датасетом kitti-360. Переместить папку `kitti360scripts` в корень проекта
   (содержит скрипты для тестирования. нельзя установить через pip)

### Подготовка данных
1. Скачать датасет [kitti-360](http://www.cvlibs.net/datasets/kitti-360/), а именно:
    - `Fisheye Images`
    - `Calibrations`
    - `Vehicle Poses`
    - `Accumulated Point Clouds for Train & Val`
    - `Test Semantic`
2. Запустить скрипт для конвертации парных fisheye в равнопромежуточные проекции
(предполагается, что изображения находятся в папках `KITTI-360/data_2d_raw/2013_05_28_drive_*_sync/image_0(2|3)/data_rgb/`)
```bash
python fisheye2equirect.py
```
Равнопромежуточные проекции будут сохранены в папках `.data_2d_equirect/2013_05_28_drive_*_sync/data_rgb/`

### Cегментация изображений
1. Веса модели `Segment Anything` должны находиться по этому пути `ckpt/sam_vit_h_4b8939.pth`
2. Запустить скрипт для сегментации изображений
```bash
python segment2d.py
```
Результаты будут сохранены в папке `data_2d_semantics/`

