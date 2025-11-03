from django.shortcuts import render
import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY

def chat_view(request):
    response_text = ""
    user_input = ""

    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            try:
                response = openai.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input},
                    ]
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                response_text = f"Error: {e}"

    return render(request, 'chat/chat.html', {
        'user_input': user_input,
        'response_text': response_text
    })
