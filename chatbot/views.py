from django.shortcuts import render
from django.http import JsonResponse
from .chatbot_logic import KnowledgeBase

knowledge_base = KnowledgeBase()

def upload_file(request):
    """Handle file uploads and process datasets to train the knowledge base."""
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type in ['txt', 'csv', 'json']:
            knowledge_base.add_data(uploaded_file, file_type)
            return JsonResponse({'message': 'File uploaded and knowledge base updated.'})
        else:
            return JsonResponse({'message': 'Unsupported file format.'}, status=400)

    return render(request, 'upload.html')

def chatbot_view(request):
    """Handle user queries by querying the knowledge base."""
    if request.method == 'POST':
        user_input = request.POST.get('message')
        response = knowledge_base.query(user_input)
        return JsonResponse({'response': response})

    return render(request, 'chat.html')
