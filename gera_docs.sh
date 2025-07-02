#!/bin/bash
cd docs/

sphinx-apidoc -o source/ ../GeoLightning

make html