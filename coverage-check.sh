#/usr/bin/bash

coverage run -m --source=eggplant unittest discover
coverage report
coverage xml
git add coverage.xml
