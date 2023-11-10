# clusterxplain
Provide feature relevance scores fo clustering using various explanation methods.

## Installation
Clusterxplain can be easily installed via pip:
```
pip install clusterxplain
```
The source coe is available at: https://github.com/AlexF1994/cluster-explain

## How to use it
The official documentation can be found on https://cluster-explain.readthedocs.io/en/latest/index.html.

But you can also jump right to the quickstart [notebook](docs/source/Quickstart.ipynb).

## Contributions
All contributions, bug reports, bug fixes, documentation improvements, enhancements, ideas etc. are very welcome.

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

    pip install -U pre-commit
    pre-commit install

We use `pytest` as test framework. To execute the tests, please run

    poetry run pytest tests

To run the tests with coverage information, please use

    poetry run pytest tests --cov=src --cov-report=xml

and have a look at the `htmlcov` folder, after the tests are done.

## Contact

Alexander Fottner (alexander.fottner@uni-a.de)

## License
[MIT](LICENSE)
