# cxplain

Provide feature relevance scores fo clustering.

## Getting Started

To set up your local development environment, run:

    poetry install

Behind the scenes, this creates a virtual environment and installs cxplain along with its dependencies (like `pip install -e .`) into that virtualenv. Whenever you run `poetry run <command>`, that `<command>` is actually run inside the virtualenv managed by poetry.

You can now import functions and classes from the module with `import cxplain`.

### Testing

We use `pytest` as test framework. To execute the tests, please run

    poetry run pytest tests

To run the tests with coverage information, please use

    poetry run pytest tests --cov=src --cov-report=xml

and have a look at the `htmlcov` folder, after the tests are done.

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks (`notebooks/`) without running into import errors by running:

    poetry run jupyter notebook

or

    poetry run jupyter-lab

This starts the jupyter server inside the project's virtualenv.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

    poetry add ipykernel
    poetry run python -m ipykernel install --user --name="cxplain"

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Distribution Package

To build a distribution package (wheel), please use

    poetry build

this will clean up the build folder and then run the `bdist_wheel` command.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

    pip install -U pre-commit
    pre-commit install

## Contact

Alexander Fottner (alexander.fottner@uni-a.de)

## License

© University of Augsburg
