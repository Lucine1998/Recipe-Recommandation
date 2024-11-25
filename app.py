import gradio as gr
from rag import similarity_search, ask_question_with_context, get_prompt  # Import functions from rag.py
from YOLO import YOLOProcessor  # Import the YOLOProcessor class from YOLO.py
import os
import requests  # For making HTTP requests to the local API

# Initialize the YOLO model globally when the app starts
yolo_processor = YOLOProcessor(weights_path="best.pt")

# Session history to keep track of conversations for different users
session_history = {}

def process_input(user_input, image):
    """
    Handles text input and optional image upload. Calls similarity_search and ask_question_with_context.
    If an image is uploaded, processes the image with YOLO and generates a prompt.
    """
    if image or user_input:
        # Create the user message (with image if provided)
        user_message = {"role": "user", "content": user_input or ""}
        if image:
            # Convert image to a format suitable for display
            user_message["image"] = image

        # Process the image with YOLO and get the prompt
        image_info = None
        if image:
            image_info = yolo_processor.process(image)
        prompt = get_prompt(user_input, image_info)

        # Perform the AI task and generate the response
        response = ask_question_with_context(prompt, [])
        assistant_message = {"role": "assistant", "content": response}
        return [user_message, assistant_message], "", None  # Reset user_input and image fields
    
    # If no valid input is provided
    return [{"role": "assistant", "content": "Please provide a message or an image."}], "", None

def call_local_llama(user_input, image):
    """
    Handles text input and optional image upload for the LocaLlama API call.
    """
    # Generate or retrieve session_id for the user (mocked with chat_history for simplicity)
    session_id = "current_session"  # Replace with actual session management logic if needed
    if session_id not in session_history:
        session_history[session_id] = []  # Initialize session history for a new session
    
    # Add user message to session history
    user_message = {"role": "user", "content": user_input or "Image uploaded for recipe suggestion."}
    if image:
        user_message["image"] = image  # Optionally add image to the message
    session_history[session_id].append(user_message)

    # Process the image with YOLO and get the prompt
    image_info = None
    if image:
        image_info = yolo_processor.process(image)
    prompt = get_prompt(user_input, image_info) if user_input or image else "No query provided."

    # Construct the messages list to include the session history (last 5 messages for context)
    messages = [{"role": "user", "content": prompt}]
    
    # Add the last 5 messages from session history to context (both user and assistant messages)
    conversation_context = session_history[session_id][-5:]  # Get the last 5 messages
    for msg in conversation_context:
        messages.append({"role": msg['role'], "content": msg['content']})
    
    # Make the POST request to the Llama API with the updated messages
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:3b",  # Model version or identifier
                "messages": messages,  # Include the full conversation history
                "stream": False  # Set to True if you want streaming responses
            },
            timeout=3000
        )
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", {})
            content = message.get("content", "No content received.")
            assistant_message = {"role": "assistant", "content": content}
            
            # Append the assistant's response to session history
            session_history[session_id].append(assistant_message)

            return [user_message, assistant_message], "", None  # Return response
        else:
            return [{"role": "assistant", "content": f"Error: {response.status_code} - {response.text}"}], "", None
    except Exception as e:
        return [{"role": "assistant", "content": f"Error communicating with LocaLlama API: {str(e)}"}], "", None
    
    # If no valid input is provided
    return [{"role": "assistant", "content": "Please provide a message or an image."}], "", None

def display_image():
    """Function to display the image below the 'More Details' button"""
    return gr.Image("result.jpg", label="Annotated Image", visible=True)

# Front-End Layout using Gradio
with gr.Blocks(css="""
    body {
        background-color: #F7F7F7;  /* Light Grayish White */
    }
    #header {
        text-align: center;
        color: white;
        background-color: #E8CCC9;
        padding: 15px;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-content {
        background-color: #FFFFFF;  /* Pure White for contrast */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: row;
    }
    #user_input {
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        border: 2px solid #50E3C2; /* Teal Green */
        font-size: 1rem;
        color: #4A4A4A; /* Dark Gray */
    }
    #send_btn {
        background-color: #4A90E2; /* Soft Blue */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        border: none;
        cursor: pointer;
    }
    #send_btn:hover {
        background-color: #357ABD; /* Darker Blue on hover */
    }
    #local_btn {
        background-color: #50E3C2; /* Teal Green */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        border: none;
        cursor: pointer;
    }
    #local_btn:hover {
        background-color: #3CB29D; /* Darker Green on hover */
    }
    
    .chatbox {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px;
        background-color: #FFFFFF; /* White */
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    #instructions {
        font-size: 0.9rem;
        color: #4A4A4A; /* Dark Gray */
        background-color: #F7F7F7; /* Light Grayish White */
        padding: 15px;
        border-radius: 8px;
    }
    #instructions b {
        color: #4A90E2; /* Soft Blue */
    }
    #results_image {
        max-width: 100%;  
        height: auto;  
    }
    .left-column {
        flex: 5;  
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .right-column {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
""") as demo:
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            # 🍳 MealMate  
            ### Your AI Assistant for Recipe Suggestions  
            Ask questions or upload an image to get personalized recipe recommendations!  
            """,
            elem_id="header"
        )
    
    # Main layout
    with gr.Row(elem_classes=["main-content"]):
        with gr.Column(scale=1, min_width=400, elem_classes=["right-column"]):  # All input buttons on the right side
            with gr.Row():
                image_upload = gr.Image(label="Upload an Image (Optional)", type="pil")
            user_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message",
                elem_id="user_input",
                lines=1,
                max_lines=3
            )
            submit_btn = gr.Button("Submit", elem_id="send_btn")
            model_choice = gr.Checkbox(label="Use LocaLlama", value=False)
            more_details_btn = gr.Button("More Details", elem_id="more_details_btn")
            results_image = gr.Image(label="Annotated Image", elem_id="results_image", visible=False)
        
        with gr.Column(scale=2, min_width=600, elem_classes=["left-column"]):  
            chatbot = gr.Chatbot([], label="Chatbot", elem_classes=["chatbox"], type="messages")
            gr.Markdown(
                """
                *How it works:*  
                1. Type your message or provide ingredients.  
                2. Submit your query to receive recipe suggestions.  
                3. The chatbot will process your input and provide results.  
                """,
                elem_id="instructions"
            )

    # Define interactions
    submit_btn.click(
        lambda user_input, image, use_local_llama: call_local_llama(user_input, image) if use_local_llama else process_input(user_input, image),
        inputs=[user_input, image_upload, model_choice],
        outputs=[chatbot, user_input, image_upload],
    )

    more_details_btn.click(
        display_image,
        inputs=[],
        outputs=[results_image],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
