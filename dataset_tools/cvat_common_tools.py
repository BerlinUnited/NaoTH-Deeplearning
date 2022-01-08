import json
import requests
import toml


def login(session):
    # TODO document credential file
    # TODO use main config toml instead of credential file (maybe)
    try:
        with open('.credentials') as f:
            secrets = json.load(f)
    except Exception as err:
        raise SystemExit(err)

    login_data = {'username': secrets["name"], 'password': secrets["pass"]}
    try:
        response = session.post('https://ball.informatik.hu-berlin.de/api/v1/auth/login', data=login_data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def get_data_root():
    with open('../config.toml', 'r') as f:
        config_dict = toml.load(f)

    return config_dict["data_root"]
