### Setting up python environment
1. ```pyenv install 3.11.9```
2. ```poetry local 3.11.9```
3. ```poetry init -n```
4. ```poetry config virtualenvs.in-project true```
5. ```poetry env use $(pyenv which python)```
6. ```poetry install```
