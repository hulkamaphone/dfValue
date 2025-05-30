# About

dfValue is a python notebook for aggregating and providing financial stock information.


## Installation

The notebook is built on Jupyter notebook with the [Anaconda](https://anaconda.org/) python distribution in mind.

## Usage

Some key packages will need to be installed that may not already exist in the default environment (especially a base Docker image like Jupyter's [minimal-notebook](https://hub.docker.com/r/jupyter/minimal-notebook)). Note the below code gest run in the notebook file anyway.

```python
!pip install yfinance html5lib --q
```

Most of the packages will be available in Anaconda environment and above step likely not required.


## Limitations
For the most part, the process is `not` idempotent. Future versions should resolve this as it gets migrated into an application-type workflow.


## License

[MIT](https://choosealicense.com/licenses/mit/)