## **📚 Machine Learning Chatbot (Django + React)**

### **Introduction**
This is a **machine learning-based chatbot** built using **Django** (backend) and **React** (frontend), allowing users to upload datasets (TXT, CSV, JSON), train a **Naive Bayes machine learning model**, and query the trained model in real-time. The chatbot uses **WebSockets** for real-time communication, allowing for dynamic user interactions, while the knowledge base is created using user-uploaded data.

This project provides:
- **Dataset ingestion** from text files, CSVs, and JSON.
- **Dynamic machine learning model training** to create an intelligent chatbot.
- **Real-time chat** using Django Channels and React.
- A simple **frontend interface** allowing for file uploads and chat.

---

## **Features**
1. **📂 File Upload**: Upload datasets to train the chatbot.
2. **📊 Real-Time Chat**: Chat with the bot in real time, with responses based on the trained model.
3. **⚡ Machine Learning Model**: Dynamic training using **TF-IDF** and **Naive Bayes**.
4. **🎤 Speech Recognition** (Optional): Speak to the bot using the browser’s Web Speech API.
5. **🔔 Notifications**: Get browser notifications when new messages arrive.
6. **💻 WebSockets**: Real-time two-way communication for instant responses.

---

## **Project Structure**

```plaintext
chatbot_project/
├── backend/                        # Django Backend
│   ├── chatbot_project/             # Django project config
│   │   ├── asgi.py                  # ASGI config for WebSockets
│   │   ├── settings.py              # Django settings, WebSocket config
│   │   ├── urls.py                  # Main URL routing for HTTP views and file upload
│   └── chatbot/                     # Main chatbot app
│       ├── consumers.py             # WebSocket consumer for real-time chat
│       ├── routing.py               # WebSocket URL routing
│       ├── views.py                 # Django views for file upload and chatbot queries
│       ├── chatbot_logic.py         # Chatbot logic: file processing, model training, queries
│       └── intents.json             # Static intents data (optional, depending on the use case)
├── frontend/                        # React Frontend
│   ├── public/                      # Public files (HTML, assets)
│   │   └── index.html               # Main HTML file
│   ├── src/                         # React source files
│   │   ├── components/              # React components
│   │   │   └── Chat.js              # Main chat component (file upload, WebSocket)
│   │   ├── App.js                   # Main React App component
│   │   ├── index.js                 # Entry point for React
│   │   └── Chat.css                 # Styling for chat interface
├── Procfile                         # Heroku deployment configuration
├── requirements.txt                 # Python dependencies (Django, channels, joblib, pandas, etc.)
└── README.md                        # Project documentation
```

---

## **Installation & Setup**

### 1. **Backend Setup (Django)**
```bash
# Clone the repository
git clone https://github.com/kyleBrian/machine-learning-chatbot.git
cd machine-learning-chatbot/backend/

# Create a virtual environment and install dependencies
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Apply migrations and run the Django server
python manage.py migrate
python manage.py runserver
```

### 2. **Frontend Setup (React)**
```bash
# Navigate to the frontend folder
cd ../frontend/

# Install React dependencies
npm install

# Start the React frontend
npm start
```

---

## **Usage**

### **Uploading Datasets**

To train the chatbot, upload datasets via the provided interface on the frontend. The supported file formats are **TXT**, **CSV**, and **JSON**. The dataset will be processed and used to train the machine learning model.

#### **File Upload in Chat.js**

```jsx
const handleFileUpload = async () => {
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post('/upload/', formData);
      alert(response.data.message);  // Show upload result
    } catch (error) {
      console.error('File upload failed:', error);
    }
  }
};
```

### **Chatting with the Bot**

Once the dataset is uploaded and processed, you can chat with the bot in real-time via WebSockets. The responses are generated based on the trained model.

#### **Real-Time Communication via WebSocket**

```jsx
const handleSendMessage = () => {
  if (inputMessage.trim() !== '') {
    sendMessage(JSON.stringify({ message: inputMessage }));  // Send message to backend
    setInputMessage('');  // Clear the input field
  }
};
```

---

## **Deployment**

### **Deploying the Backend (Django) to Heroku**

1. **Install Heroku CLI**:
   ```bash
   heroku login
   ```

2. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

3. **Set up the `Procfile`** for Heroku (to use `daphne` for WebSocket support):
   ```plaintext
   web: daphne chatbot_project.asgi:application --port $PORT --bind 0.0.0.0
   ```

4. **Push the Backend to Heroku**:
   ```bash
   git push heroku master
   ```

### **Deploying the Frontend (React) to Netlify**

1. **Build the React App**:
   ```bash
   npm run build
   ```

2. **Deploy to Netlify**:
   - You can drag and drop the **`build/`** folder onto [Netlify](https://www.netlify.com) for easy deployment.

---

## **File Upload and Data Processing Flow**

- **Backend (Django)**: When a file is uploaded, Django processes the file, passes the data to the **Naive Bayes model** (with **TF-IDF** for vectorization), and saves the trained model.
  
  ```python
  def add_data(self, file_data, file_type):
      if file_type == 'txt':
          data = [line.strip() for line in file_data if line.strip()]
          self.train_model(data)
      elif file_type == 'csv':
          df = pd.read_csv(file_data)
          self.train_model(df['question'], df['answer'])
      elif file_type == 'json':
          data = json.load(file_data)
          self.train_model(questions, answers)
  ```

---

## **Technologies Used**
- **Django**: Backend framework.
- **React**: Frontend framework.
- **Django Channels**: WebSockets for real-time communication.
- **Scikit-learn**: Machine learning library (TF-IDF + Naive Bayes).
- **Heroku**: Deployment platform for the backend.
- **Netlify**: Deployment platform for the frontend.
- **Redis**: WebSocket backend (used by Django Channels).

---


## **Developer Info**

- **GitHub**: [kyleBrian](https://github.com/kyleBrian)
- **Email**: [kylabelma@gmail.com](mailto:kylabelma@gmail.com)
- **Phone**: +254 758034649

---



## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
