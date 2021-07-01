IPython's magic commands, now implemented in pure Python (Python using only built-in libraries).

How to install using `pip` in Windows 10:<br/>
1. Open command prompt (`Win+R`, type in 'cmd')
2. Type command `pip install magic-commands`<br/>
2.1. To install in Python version `X.Y`, do `pipX.Y install magic-commands` or `py -X.Y -m pip install magic-commands` (requires global py launcher)<br/>
2.2. If `pip` is not already installed, do `py -m ensurepip`, or in Python version `X.Y` do `py -X.Y -m ensurepip`<br/>
2.3. If neither `py` nor `python` is a command, make sure to have installed Python with the `Add PythonX.Y to PATH` and `global py launcher` options checked.<br/>

How to install using `pip` in Ubuntu 20.04 LTS (Needs confirmation):<br/>
1. Open Terminal<br/>
2. Run `pip install magic-commands`<br/>
2.1. To install in Python version 'X.Y', do `pipX.Y install magic-commands` or `pythonX.Y -m pip install magic-commands`<br/>
2.2. If `pip` is not already installed, do `python -m ensurepip`, or in Python version `X.Y` do `pythonX.Y -m ensurepip` OR `sudo apt install python-pip`<br/>
2.3. If neither `python` nor `python3` is a command, Python can be installed using `sudo apt install python`, or specifically Python version `X.Y` do<br/>
    ```
    sudo apt update
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install pythonX.Y
    ```

To check if `magic-commands` is installed:<br/>
1. On Windows 10<br/>
1.1 Open command prompt (`Win+R`, type in 'cmd')<br/>
1.2 Open python by entering command `py` or `python`. To open Python version `X.Y` specifically, do `py -X.Y`<br/>
1.3 Type in `import magiccmds`. If no `ImportError` occurs, the package is installed.<br/>
2. On Ubuntu 20.04 LTS<br/>
2.1 Open Terminal<br/>
2.2 Open python by entering command `python` or `python3`. To open Python version `X.Y` specifically, do `pythonX.Y`<br/>
2.3 Type in `import magiccmds`. If no `ImportError` occurs, the package is installed.
