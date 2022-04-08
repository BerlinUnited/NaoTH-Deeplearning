from pathlib import Path
import requests
import toml
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError


def cvat_login(session):
    try:
        with open('../config.toml', 'r') as f:
            config_dict = toml.load(f)

            username = config_dict["cvat_user"]
            password = config_dict["cvat_pass"]
    except Exception as err:
        print("Could not get credentials from toml file")
        raise SystemExit(err)

    login_data = {'username': username, 'password': password}
    try:
        response = session.post('https://ball.informatik.hu-berlin.de/api/auth/login', data=login_data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    # CVAT requires that each post/patch/delete call sends the authorization token as well. 
    # This is a weird mix of token and session cookie authorization
    # Here we put the cvat token inside the session cookie so that later retrieval is easier
    my_cookie = {"name": "token", "value": response.json()["key"]}
    session.cookies.set(**my_cookie)


def get_data_root():
    config_path = Path(__file__).parent.parent.resolve() / "config.toml"
    with open(str(config_path), 'r') as f:
        config_dict = toml.load(f)

    return config_dict["data_root"]


def get_logs_root():
    config_path = Path(__file__).parent.parent.resolve() / "config.toml"
    with open(str(config_path), 'r') as f:
        config_dict = toml.load(f)

    return config_dict["logs_root"]


def download_function(origin, target):
    def dl_progress(count, block_size, total_size):
        print('\r', 'Progress: {0:.2%}'.format(min((count * block_size) / total_size, 1.0)), sep='', end='', flush=True)

    if not Path(target).exists():
        target_folder = Path(target).parent
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        return

    error_msg = 'URL fetch failure on {} : {} -- {}'
    try:
        try:
            urlretrieve(origin, target, dl_progress)
            print('\nFinished')
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.reason))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if Path(target).exists():
            Path(target).unlink()
        raise
