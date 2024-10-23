import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .chatbot_logic import KnowledgeBase

# Initialize knowledge base
knowledge_base = KnowledgeBase()

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        """Handle receiving messages via WebSocket."""
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Query the knowledge base and get a response
        response = knowledge_base.query(message)

        # Send response back to WebSocket
        await self.send(text_data=json.dumps({
            'message': response
        }))
