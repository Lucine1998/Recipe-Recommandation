import gradio as gr
from rag import similarity_search, ask_question_with_context  # Import functions from rag.py
from YOLO import YOLOProcessor  # Import the YOLOProcessor class from YOLO.py
import os
import requests  # For making HTTP requests to the local API

# Initialize the YOLO model globally when the app starts
yolo_processor = YOLOProcessor(weights_path="best.pt")

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
        if image:
            prompt = yolo_processor.process(image)
            prompt += f"\nUser asked: {user_input}"  # Append the user's question to the prompt
        else:
            prompt = user_input

        # Perform similarity search and format the response
        search_results = similarity_search(prompt, top_k=5)
        formatted_context = [
            f"Recipe ID: {res['id']}, Name: {res['name']}, Description: {res['description']}"
            for res in search_results
        ]
        question = "Can you suggest the most nutritious option among these recipes?"
        response = ask_question_with_context(question, formatted_context)

        assistant_message = {"role": "assistant", "content": response}
        return [user_message, assistant_message], "", None  # Reset user_input and image fields
    
    # If no valid input is provided
    return [{"role": "assistant", "content": "Please provide a message or an image."}], "", None

def call_local_llama(user_input, image):
    """
    Handles text input and optional image upload for the LocaLlama API call.
    """
    if image or user_input:
        # Create the user message (with image if provided)
        user_message = {"role": "user", "content": user_input or ""}
        if image:
            # Convert image to a format suitable for display
            user_message["image"] = image

        # Process the image with YOLO and get the prompt
        if image:
            prompt = yolo_processor.process(image)
            prompt += f"\nUser asked: {user_input}"  # Append the user's question to the prompt
        else:
            prompt = user_input

        # Make a POST request to the LocaLlama API
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2:3b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=3000
            )
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "No response received.")
                content = message.get("content", "No content received.")
                assistant_message = {"role": "assistant", "content": content}
                return [user_message, assistant_message], "", None  # Reset user_input and image fields
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
        background-color: #ffffff;  
    }
    #header {
        text-align: center;
        color: white;
        background-color: #007BFF;
        padding: 15px;
        font-size: 2rem;
        margin-bottom: 20px;
        border-radius: 10px;
    }
    .main-content {
        background-color: #f0f8ff;  
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
        border: 2px solid #007BFF;
        font-size: 1rem;
    }
    #send_btn {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        border: none;
        cursor: pointer;
    }
    #send_btn:hover {
        background-color: #218838;
    }
    #local_btn {
        background-color: #ff5722;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        border: none;
        cursor: pointer;
    }
    #local_btn:hover {
        background-color: #e64a19;
    }
    
    .chatbox {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
        max_height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        # flex: 1;
    }
    #instructions {
        font-size: 0.9rem;
        color: #555;
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
    }
    #instructions b {
        color: #007BFF;
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
    # .right-column {
    #     flex: 1;
    #     display: flex;
    #     flex-direction: column;
    # }
""") as demo:
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            # 🤖 Chatbot with Recipe Suggestions  
            A simple chatbot interface with recipe suggestions based on your input.  
            Ask a question or provide ingredients to get recipe recommendations.  
            """,
            elem_id="header"
        )
    
    # Main layout with light blue background
    with gr.Row(elem_classes=["main-content"]):
        with gr.Column(scale=1, min_width=400, elem_classes=["right-column"]):  # All input buttons on the right side
            # Input area for message, buttons, and image upload
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
            # Add a button to show the "More Details" (image) below the input
            results_image = gr.Image(label="Annotated Image", elem_id="results_image", visible=False)
        
        with gr.Column(scale=2, min_width=600, elem_classes=["left-column"]):  
            # Chatbox area with light blue background
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
        outputs=[chatbot, user_input, image_upload],  # Reset text and image inputs
    )

    # Define the interaction for the "More Details" button to display the image below
    more_details_btn.click(
        display_image,
        inputs=[],  # No inputs needed to show the image
        outputs=[results_image],  # Show the image in the specified output element
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
