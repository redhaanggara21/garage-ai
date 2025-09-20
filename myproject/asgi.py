import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from django.urls import path
from classifier.consumers import ProgressConsumer

os.environ.setdefault("DJANGO_SETTINGS_MODULE","myproject.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/progress/",ProgressConsumer.as_asgi()),
        ])
    )
})
