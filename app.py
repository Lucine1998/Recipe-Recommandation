import gradio as gr
from rag import similarity_search, ask_question_with_context  # Import functions from rag.py
from YOLO import YOLOProcessor  # Import the YOLOProcessor class from YOLO.py

# Initialize the YOLO model globally when the app starts
yolo_processor = YOLOProcessor(weights_path="best.pt")

def process_input(user_input, image):
    """
    Handles text input and optional image upload. Calls similarity_search and ask_question_with_context.
    If an image is uploaded, processes the image with YOLO and generates a prompt.
    """
    if image or user_input:
        # Process the image with YOLO and get the prompt
        if image:
            prompt = yolo_processor.process(image)
            prompt += f"\nUser asked: {user_input}"  # Append the user's question to the prompt
        else: 
            prompt = user_input
        # If no image is uploaded, perform the similarity search based on the user's input
        search_results = similarity_search(prompt, top_k=5)

        # Format the search results into a string that the LLM can use for context
        formatted_context = [
            f"Recipe ID: {res['id']}, Name: {res['name']}, Description: {res['description']}"
            for res in search_results
        ]
        # Ask the LLM with the formatted context and the user's input as the question
        question = "Can you suggest the most nutritious option among these recipes?"
        response = ask_question_with_context(question, formatted_context)
        return [{"role": "user", "content": user_input}, {"role": "assistant", "content": response}]
    
    # If no valid input is provided
    return [{"role": "assistant", "content": "Please provide a message or an image."}]

# Front-End Layout using Gradio
with gr.Blocks(css="""
    body {
        background-color: #f8f9fa;  /* Light gray for subtle background */
        font-family: 'Arial', sans-serif;
    }
    #header {
        text-align: center;
        color: #ffffff;
        background-color: #4caf50;  /* A pleasing green tone for the header */
        padding: 20px;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 20px;
        border-radius: 12px;
    }
    .main-content {
        background-color: #ffffff;  /* Pure white for the main area */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        margin: 0 10px;
    }
    #user_input {
        width: 100%;
        border-radius: 12px;
        padding: 12px;
        border: 2px solid #4caf50;
        font-size: 1rem;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    #send_btn {
        background-color: #007bff; /* A vibrant blue */
        color: #ffffff;
        font-weight: bold;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    #send_btn:hover {
        background-color: #0056b3;
    }
    .chatbox {
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 12px;
        background-color: #f9f9f9;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    #instructions {
        font-size: 1rem;
        color: #495057;
        background-color: #e9ecef; /* Soft gray for the instruction area */
        padding: 20px;
        border-radius: 12px;
    }
    #instructions b {
        color: #4caf50;
    }
""") as demo:
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            # 🍴 Chatbot with Recipe Suggestions  
            A sleek chatbot interface for personalized recipe suggestions.  
            Just type a message or upload ingredients to get started!  
            """,
            elem_id="header"
        )
    
    # Main layout with a clean white card
    with gr.Row(elem_classes=["main-content"]):
        with gr.Column(scale=2, min_width=600):
            # Chatbox area
            chatbot = gr.Chatbot([], label="Chatbot", elem_classes=["chatbox"], type="messages")

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your query here...",
                    label="Your Message",
                    elem_id="user_input",
                    lines=1,
                    max_lines=3
                )
                submit_btn = gr.Button("Send", elem_id="send_btn")

            with gr.Row():
                image_upload = gr.Image(label="Upload an Image (Optional)", type="pil")
        
        with gr.Column(scale=1, min_width=300):
            gr.Markdown(
                """
                **How to Use:**  
                1. Enter ingredients or a question.  
                2. Click "Send" to process your query.  
                3. Receive personalized recipe suggestions!  
                """,
                elem_id="instructions"
            )
    
    # Define interactions
    submit_btn.click(
        process_input,
        inputs=[user_input, image_upload],
        outputs=[chatbot],
    )


# Launch the app
if __name__ == "__main__":
    demo.launch()
