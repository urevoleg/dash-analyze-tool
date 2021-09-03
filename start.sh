#!/bin/bash
cd /home/dead/iotviewer/
venv/bin/gunicorn wsgi:app -b 0.0.0.0:8050 --reload