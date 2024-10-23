import React, { useState, useEffect } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import axios from 'axios';

const WS_URL = 'ws://localhost:8000/ws/chat/';

const Chat = () => {
  const [messageHistory, setMessageHistory] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [file, setFile] = useState(null);

  const { sendMessage, lastMessage } = useWebSocket(WS_URL);

  useEffect(() => {
    if (lastMessage !== null) {
      const response = JSON.parse(lastMessage.data);
      setMessageHistory((prev) => [...prev, { type: 'bot', message: response.message }]);
    }
  }, [lastMessage]);

  const handleSendMessage = () => {
    if (inputMessage.trim() !== '') {
      setMessageHistory((prev) => [...prev, { type: 'user', message: inputMessage }]);
      sendMessage(JSON.stringify({ message: inputMessage }));
      setInputMessage('');
    }
  };

  const handleFileUpload = async () => {
    if (file) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post('/upload/', formData);
        alert(response.data.message);
      } catch (error) {
        console.error('File upload failed:', error);
      }
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {messageHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            {msg.message}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Type a message..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>

      <div className="file-upload">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleFileUpload}>Upload Dataset</button>
      </div>
    </div>
  );
};

export default Chat;
