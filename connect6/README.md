# connect6
A Python library for developing Connect6 AI algorithms


## Install
```bash
$ pip install connect6
```


## Usage
Play
```bash
$ python -m connect6
```
Environment
```python
import connect6

env = connect6.Connect6()
for move in  [180, 179, 161, 160, 200, 181, 199, 140, 220, 198, 162, 120]:
    env(move)
print(env)
```

### License
MIT