This folder contains code for predicting the position of robots in images provided by Visual Analatics API
The goal is to use a YOLO model for annotating robocup image data.

## How to setup direnv
It's recommended to set up a virtual enviroment for this directory



```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
sudo apt install direnv
```

add this to your .bashrc
if you use other shells see https://direnv.net/docs/hook.html

```bash
eval "$(direnv hook bash)"
```

create a .envrc file in this folder 

```bash
export VAT_API_URL=http://localhost:8000/
export VAT_API_TOKEN=<your-token-here>

source ./venv/bin/activate
unset PS1
```

after every change of .envrc you need to run
```bash
direnv allow
```
