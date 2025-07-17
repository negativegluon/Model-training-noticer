import requests
import json
import yaml

def load_config(config_path="message_handler_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

def send_error_to_port(error_message):
    """
    通过 HTTP POST 向 napcat 服务器发送私聊消息
    """
    # napcat 服务器接口地址
    napcat_server_url = config['napcat_config']['napcat_server_url']
    url = napcat_server_url.rstrip('/') + "/send_private_msg"

    qq_id = config["message_handler_config"]["target_qq"]

    payload = json.dumps({
        "user_id": qq_id,
        "message": [
            {
                "type": "text",
                "data": {
                    "text": error_message
                }
            }
        ],
        "token": config['napcat_config']['napcat_server_token'],
    })
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post(url, headers=headers, data=payload)
    print(response.text)