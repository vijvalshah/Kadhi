"""Global constants and defaults."""

APP_NAME = "kadhi"
CONFIG_FILE = "kadhi.yaml"
KADHI_DIR = ".kadhi"
EXPERIMENTS_DB = "experiments.db"
PROJECT_URL = "https://trykadhi.dev"

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\\n' }}"
    "{% endif %}{% endfor %}"
)
