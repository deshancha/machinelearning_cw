import os
from di.container import AppContainer

# DI container
container = AppContainer()

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
container.config.from_dict({
    "data": {
        "data_dir": os.path.join(base_dir, "data")
    },
    "models": {
        "model_dir": os.path.join(base_dir, "outputs/models")
    }
})

server = container.api_server()
api = server.api
