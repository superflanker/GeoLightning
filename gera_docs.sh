#!/bin/bash
source venv/bin/activate
cd docs/

sphinx-apidoc -o source/ ../GeoLightning

make html
deactivate