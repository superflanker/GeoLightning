#"/bin/bash"

sphinx-apidoc -o docs/source ./GeoLightning
cd docs/

make html