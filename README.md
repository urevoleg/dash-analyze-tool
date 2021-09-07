# dash-analyze-tool

Для запуска необходимо выполнить:

0. Перейти в папку проекта: `cd dash-analyze-tool`. Далее все команды выполняются в папке проекта
1. `python3 venv venv`
2. `source venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. Укажите директорию с датасетами (для приложений `eda` и `reduce`) - в файле `flask.cfg` укажите путь для переменной `DATASETDIR`.
Для приложения `explore` выбор датасета происходит из окна браузера.
По-умолчанию, приложение использует `memcached` для кеширования вычисленных результатов `reduce` (для некоторых параметров и данных метод `UMAP` работает долго). 
Для кеша требуется установленный `memcached` c параметрами по-умолчанию. Чтобы отключить использование кеша установите опцию `IS_MEMCACHED=False` в файле `flask.cfg`.
При такой настройке `IS_MEMCACHED=False` в качестве кеша используется словарь.
5. Запустите: `gunicorn wsgi:app -b 0.0.0.0:8050 -t 0`