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


# temp testing:

```
python DFReader.py "2023-11-10 13-55-14.BIN"
# wait 30s -> ctrl+C
```

generates "2023-11-10 13-55-14.BIN.bin.out"
