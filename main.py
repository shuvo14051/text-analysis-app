import streamlit as st
from transformers import pipeline

# Initialize the sentiment analysis and NER pipelines
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner = pipeline("ner", grouped_entities=True)

# Set the page configuration
st.set_page_config(
    page_title="Text Analysis App",
    page_icon="🔍",
    layout="centered",
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Named Entity Recognition"])

# Home Page
if page == "Home":
    st.title("Welcome to the Text Analysis App")
    st.write("""
        This app provides two main functionalities:
        - **Sentiment Analysis**: Analyze the sentiment of the text you enter.
        - **Named Entity Recognition (NER)**: Identify and highlight entities such as people, organizations, and locations in the text.
        
        Use the navigation menu on the left to switch between different functionalities.
    """)

# Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    user_input = st.text_area("Enter text to analyze sentiment", "")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            result = classifier(user_input)
            label = result[0]['label']
            score = result[0]['score']
            st.write(f"Sentiment: **{label}** with a confidence score of **{score:.2f}**")
        else:
            st.write("Please enter some text to analyze.")

# Named Entity Recognition Page
elif page == "Named Entity Recognition":
    st.title("Named Entity Recognition (NER)")
    
    colored_boxes_html = """
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="background-color: red; color: white; width: 80px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
            Person
        </div>
        <div style="background-color: green; color: white; width: 90px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
            Location
        </div>
        <div style="background-color: blue; color: white; width: 120px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
            Organization
        </div>
    </div>
    """
    
    st.markdown(colored_boxes_html, unsafe_allow_html=True)
    
    user_input = st.text_area("Enter text to analyze:", "")
    
    # Define color mapping for entity types
    color_map = {
        "PER": "red",
        "ORG": "blue",
        "LOC": "green"
    }
    
    def highlight_entities(text, entities):
        highlighted_text = []
        last_end = 0

        for entity in entities:
            entity_text = text[entity['start']:entity['end']]
            background_color = color_map.get(entity['entity_group'], "black")
            highlighted_text.append(text[last_end:entity['start']])
            highlighted_text.append(
                f"<span style='background-color:{background_color}; color:white; font-weight:bold; padding:2px 4px;'>{entity_text}</span>"
            )
            last_end = entity['end']

        highlighted_text.append(text[last_end:])
        return "".join(highlighted_text)
    
    if st.button("Analyze NER"):
        if user_input:
            entities = ner(user_input)
            highlighted_text = highlight_entities(user_input, entities)
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.write("Please enter some text to analyze.")