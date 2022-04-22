#/usr/bin/bash

coverage run -m --source=eggplant unittest discover
coverage report
coverage xml
coverage html
git add coverage.xml
