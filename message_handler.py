from ncatbot.core import BotClient
from ncatbot.utils.config import config as ncat_config
from ncatbot.core.element import MessageChain, Text
import yaml

def load_config(config_path="message_handler_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()

ncat_config.set_bot_uin(config["napcat_config"]["bot_qq_number"])  # 设置 bot qq 号 (必填)
ncat_config.set_ws_uri(config["napcat_config"]["napcat_server_url"])  # 设置 napcat websocket server 地址
ncat_config.set_token(config["napcat_config"]["napcat_server_token"]) # 设置 token (napcat 服务器的 token)

bot = BotClient()  # 创建 bot 实例


def send_error_to_port(error_message):
    
    message = MessageChain([Text(error_message)])

    qq_id = config["message_handler_config"]["target_qq"]  # 希望发送到的qq号

    bot.api.post_private_msg(qq=qq_id, rtf=message)
