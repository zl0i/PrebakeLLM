# Вопросы владельцу

## 2026-11-09: пропал python.exe

Около 00:50 из `C:\Users\Dmitri\AppData\Local\Programs\Python\Python312\` исчез `python.exe`. `pythonw.exe`, библиотеки и `Lib/` на месте, в `Python39` его тоже нет. Сессия его не трогала. Похоже на карантин антивируса или на работу установщика. Все скрипты замеров (`bench/*.py`, `aic.py`) запускаются через `python`, поэтому нужно восстановить: вернуть файл из карантина или выполнить «Repair» в установщике Python 3.12. Это системная установка, поэтому без вашего решения не трогаю. Временный обход: `C:\Users\Dmitri\AppData\Local\Programs\Python\Python312\pythonw.exe script.py > out.txt`.
