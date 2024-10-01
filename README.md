## Initial setup

```
python3 -m virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt
```


## run

```
./main.py -t "<xxxx>.tlog"
./main.py -t "<xxxx>.tlog" --head 50000  # parse only the first 50000 entries
```