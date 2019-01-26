# hw1
# laurel estes
# 1/10/19

[![Build Status](https://travis-ci.com/laueste/algorithms-ucsf-2019.svg?token=zRpn8zZpP78U6qyaUqA7&branch=master)](https://travis-ci.com/laueste/algorithms-ucsf-2019)

Current build status in travis is displayed above. Contents below the line in this readme are taken entirely from the example file in the ucsf-bmi-203-2017/example github repo, and reproduced here for a record of how to run the example on which hw1 was based.

_____________________________________________________________________

Example python project with testing.

## usage

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `example/__main__.py`) can be run as follows

```
python -m example
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
