# import streamlit as st
# from transformers import pipeline

# # Initialize the sentiment analysis and NER pipelines
# # classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# # ner = pipeline("ner", grouped_entities=True)

# # Set the page configuration
# st.set_page_config(
#     page_title="Text Analysis App",
#     page_icon="🔍",
#     layout="centered",
# )

# @st.cache_resource
# def load_sentiment_analysis_model():
#     return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# classifier = load_sentiment_analysis_model()

# @st.cache_resource
# def load_ner_model():
#     return pipeline("ner", grouped_entities=True)
# ner = load_ner_model()


# # Sidebar for navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Named Entity Recognition"])

# # Home Page
# if page == "Home":
#     st.title("Welcome to the Text Analysis App")
#     st.write("""
#         This app provides two main functionalities:
#         - **Sentiment Analysis**: Analyze the sentiment of the text you enter.
#         - **Named Entity Recognition (NER)**: Identify and highlight entities such as people, organizations, and locations in the text.
        
#         Use the navigation menu on the left to switch between different functionalities.
#     """)

# # Sentiment Analysis Page
# elif page == "Sentiment Analysis":
#     st.title("Sentiment Analysis")
#     user_input = st.text_area("Enter text to analyze sentiment", "")
    
#     if st.button("Analyze Sentiment"):
#         if user_input:
#             result = classifier(user_input)
#             label = result[0]['label']
#             score = result[0]['score']
#             st.write(f"Sentiment: **{label}** with a confidence score of **{score:.2f}**")
#         else:
#             st.write("Please enter some text to analyze.")

# # Named Entity Recognition Page
# elif page == "Named Entity Recognition":
#     st.title("Named Entity Recognition (NER)")
    
#     colored_boxes_html = """
#     <div style="display: flex; justify-content: space-around; margin: 20px 0;">
#         <div style="background-color: red; color: white; width: 80px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
#             Person
#         </div>
#         <div style="background-color: green; color: white; width: 90px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
#             Location
#         </div>
#         <div style="background-color: blue; color: white; width: 120px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
#             Organization
#         </div>
#     </div>
#     """
    
#     st.markdown(colored_boxes_html, unsafe_allow_html=True)
    
#     user_input = st.text_area("Enter text to analyze:", "")
    
#     # Define color mapping for entity types
#     color_map = {
#         "PER": "red",
#         "ORG": "blue",
#         "LOC": "green"
#     }
    
#     def highlight_entities(text, entities):
#         highlighted_text = []
#         last_end = 0

#         for entity in entities:
#             entity_text = text[entity['start']:entity['end']]
#             background_color = color_map.get(entity['entity_group'], "black")
#             highlighted_text.append(text[last_end:entity['start']])
#             highlighted_text.append(
#                 f"<span style='background-color:{background_color}; color:white; font-weight:bold; padding:2px 4px;'>{entity_text}</span>"
#             )
#             last_end = entity['end']

#         highlighted_text.append(text[last_end:])
#         return "".join(highlighted_text)
    
#     if st.button("Analyze NER"):
#         if user_input:
#             entities = ner(user_input)
#             highlighted_text = highlight_entities(user_input, entities)
#             st.markdown(highlighted_text, unsafe_allow_html=True)
#         else:
#             st.write("Please enter some text to analyze.")



import streamlit as st
from transformers import pipeline

# Set the page configuration
st.set_page_config(
    page_title="Text Analysis and Translation App",
    page_icon="🔍",
    layout="centered",
)

@st.cache_resource
def load_sentiment_analysis_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_sentiment_analysis_model()

@st.cache_resource
def load_ner_model():
    return pipeline("ner", grouped_entities=True)

ner = load_ner_model()

@st.cache_resource
def load_translation_models():
    bntoen = pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en")
    entobn = pipeline("translation", model="csebuetnlp/banglat5_nmt_en_bn")
    return bntoen, entobn

bntoen, entobn = load_translation_models()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Named Entity Recognition", "Translation"])

# Home Page
if page == "Home":
    st.title("Welcome to the Text Analysis and Translation App")
    st.write("""
        This app provides three main functionalities:
        - **Sentiment Analysis**: Analyze the sentiment of the text you enter.
        - **Named Entity Recognition (NER)**: Identify and highlight entities such as people, organizations, and locations in the text.
        - **Translation**: Translate text between Bengali and English.
        
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

# Translation Page
elif page == "Translation":
    st.title("Bengali-English Translation")

    st.header("Translate from Bengali to English")
    bengali_text = st.text_area("Enter Bengali text:", "")

    if st.button("Translate to English"):
        if bengali_text:
            english_translation = bntoen(bengali_text)[0]['translation_text']
            st.write("**English Translation:**")
            st.write(english_translation)
        else:
            st.write("Please enter some Bengali text to translate.")

    st.header("Translate from English to Bengali")
    english_text = st.text_area("Enter English text:", "")

    if st.button("Translate to Bengali"):
        if english_text:
            bengali_translation = entobn(english_text)[0]['translation_text']
            st.write("**Bengali Translation:**")
            st.write(bengali_translation)
        else:
            st.write("Please enter some English text to translate.")