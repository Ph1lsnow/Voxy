# Video Plus - Система улучшения качества видео
#### `NOSTYLIST | RUTUBE` `ML` `PyTorch` `GFPGAN` `Deep Learning`
***
## Описание
Система улучшения качества видео, основанная на глубоких нейронных сетях. Позволяет улучшить качество видео, увеличив разрешение и убрав шумы.

## Инструкция по запуску
1. Скачать репозиторий
2. Поместить в корневую директорию файл `creds.json`, который является ключом от сервисного аккаунта Google Platform
3. Запустить первую ячейку в файле `setup1.ipynb`, желательно запустить ещё и вторую для проверки CUDA и GPU, без которых о быстрых вычислениях не может быть и речи
4. После настройки полей, связанных с Google Drive, можно запускать `video_plus.ipynb`, в котором и происходит улучшение видео в цикле, которые берутся из Google Drive



Результаты сохраняются в папку на Google Drive, эта интеграция позволяет запускать скрипт и работать с данными с любого сервера.


_В рамках задачи, искусственно было ограничено количество `FPS` выходного потока, до `1`_
"# Voxy" 
