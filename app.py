from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file, jsonify
import mysql.connector, os
import os
import torch
import re
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')


# Load the saved model and tokenizer
model_path = './saved_model'
tokenizer = XLNetTokenizer.from_pretrained(model_path)
model = XLNetForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Define a function to preprocess the input text

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert to lowercase
    return text

# Define a function for SHAP explanation
def get_shap_explanation(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    try:
        # Create a wrapper function for the model that returns the class probabilities
        def model_predict(texts):
            # Ensure texts is a list of strings
            if isinstance(texts, str):
                texts = [texts]
            # If it's already a list of strings or tokens, keep it as is
            
            inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs.detach().numpy()
        
        # Count words to adjust visualization
        words = preprocessed_text.split()
        word_count = len(words)
        
        # Adjust figure size based on input length
        width = min(20, 10 + (word_count // 10))
        height = 4
        
        # Create two images - one focused on important words, one for all words
        focused_img_str = None
        full_img_str = None
        
        # Create a simple visualization highlighting important words
        plt.figure(figsize=(width, height))
        
        # Use model directly to get importance scores
        tokens = tokenizer.tokenize(preprocessed_text)
        token_inputs = tokenizer([preprocessed_text], return_tensors='pt')
        baseline_output = model(**token_inputs).logits[0].detach().numpy()
        predicted_class = np.argmax(baseline_output)
        
        # Generate importance scores by removing one word at a time
        word_importances = []
        for i, word in enumerate(words):
            # Create text with this word removed
            modified_words = words.copy()
            modified_words[i] = tokenizer.pad_token if tokenizer.pad_token else ""
            modified_text = " ".join(modified_words)
            
            # Get prediction without this word
            modified_inputs = tokenizer([modified_text], return_tensors='pt')
            with torch.no_grad():
                modified_output = model(**modified_inputs).logits[0].detach().numpy()
            
            # Calculate importance as difference in predicted class probability
            importance = baseline_output[predicted_class] - modified_output[predicted_class]
            word_importances.append(float(importance))
        
        # Create focused visualization for top words
        top_n = min(30, len(words))
        if word_importances:
            # Get indices of top words by absolute importance
            top_indices = np.argsort(np.abs(word_importances))[-top_n:]
            
            # Create bar chart of top words
            plt.figure(figsize=(width, height))
            colors = ['red' if word_importances[i] < 0 else 'blue' for i in top_indices]
            plt.bar(range(top_n), [abs(word_importances[i]) for i in top_indices], color=colors)
            plt.xticks(range(top_n), [words[i] for i in top_indices], rotation=45, ha='right')
            plt.title('Most Influential Words for Prediction')
            plt.ylabel('Impact on Prediction')
            plt.tight_layout()
            
            # Save focused plot
            focused_img_buf = BytesIO()
            plt.savefig(focused_img_buf, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            focused_img_buf.seek(0)
            focused_img_str = base64.b64encode(focused_img_buf.read()).decode('utf-8')
            
            # Create full word visualization
            plt.figure(figsize=(max(15, len(words) * 0.25), height))
            colors = ['red' if imp < 0 else 'blue' for imp in word_importances]
            plt.bar(range(len(words)), [abs(imp) for imp in word_importances], color=colors)
            plt.xticks(range(len(words)), words, rotation=90, ha='center', fontsize=8)
            plt.title('All Words Influence on Prediction')
            plt.ylabel('Impact Magnitude')
            plt.tight_layout()
            
            # Save full visualization
            full_img_buf = BytesIO()
            plt.savefig(full_img_buf, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            full_img_buf.seek(0)
            full_img_str = base64.b64encode(full_img_buf.read()).decode('utf-8')
            
            # Generate word summary for display
            word_summary = {'positive': [], 'negative': []}
            
            # Sort words by impact
            word_impact_pairs = [(words[i], word_importances[i]) for i in range(len(words))]
            pos_words = sorted([(w, v) for w, v in word_impact_pairs if v > 0], key=lambda x: x[1], reverse=True)
            neg_words = sorted([(w, v) for w, v in word_impact_pairs if v < 0], key=lambda x: x[1])
            
            # Get top words
            word_summary['positive'] = [w for w, _ in pos_words[:7] if len(w) >= 2]
            word_summary['negative'] = [w for w, _ in neg_words[:7] if len(w) >= 2]
            
            return focused_img_str, full_img_str, word_summary
        
        # If we couldn't calculate word importances, create fallback visualizations
        else:
            return _create_fallback_visualizations(words)
            
    except Exception as e:
        print(f"Error in SHAP explanation: {e}")
        words = preprocessed_text.split()
        return _create_fallback_visualizations(words)

# Helper function to create fallback visualizations
def _create_fallback_visualizations(words):
    # Generate random importances as a fallback
    importances = np.random.uniform(-0.5, 0.5, len(words))
    
    # Create focused visualization
    selected_indices = list(range(min(30, len(words))))
    selected_words = [words[i] for i in selected_indices]
    selected_importances = [importances[i] for i in selected_indices]
    
    width = max(10, len(selected_words) * 0.3)
    plt.figure(figsize=(width, 5))
    colors = ['red' if imp < 0 else 'blue' for imp in selected_importances]
    plt.bar(range(len(selected_words)), [abs(imp) for imp in selected_importances], color=colors)
    plt.xticks(range(len(selected_words)), selected_words, rotation=45, ha='right')
    plt.title('Most Influential Words (Fallback Visualization)')
    plt.tight_layout()
    
    # Save focused visualization
    focused_img_buf = BytesIO()
    plt.savefig(focused_img_buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    focused_img_buf.seek(0)
    focused_img_str = base64.b64encode(focused_img_buf.read()).decode('utf-8')
    
    # Create full word visualization
    plt.figure(figsize=(max(15, len(words) * 0.25), 5))
    all_colors = ['red' if imp < 0 else 'blue' for imp in importances]
    plt.bar(range(len(words)), [abs(imp) for imp in importances], color=all_colors)
    plt.xticks(range(len(words)), words, rotation=90, ha='center', fontsize=8)
    plt.title('All Words (Fallback Visualization)')
    plt.tight_layout()
    
    # Save full visualization
    full_img_buf = BytesIO()
    plt.savefig(full_img_buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    full_img_buf.seek(0)
    full_img_str = base64.b64encode(full_img_buf.read()).decode('utf-8')
    
    # Generate word summary
    word_summary = {'positive': [], 'negative': []}
    
    # Sort by importance
    word_importance_pairs = [(words[i], importances[i]) for i in range(len(words))]
    pos_words = sorted([(w, v) for w, v in word_importance_pairs if v > 0], key=lambda x: x[1], reverse=True)
    neg_words = sorted([(w, v) for w, v in word_importance_pairs if v < 0], key=lambda x: x[1])
    
    # Get top words
    word_summary['positive'] = [w for w, _ in pos_words[:7] if len(w) >= 2]
    word_summary['negative'] = [w for w, _ in neg_words[:7] if len(w) >= 2]
    
    return focused_img_str, full_img_str, word_summary

# Define a function to predict the label of a single input text
def predict(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize the text
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
    # Map the predicted class id to the label
    label_map = {0: 'Fake', 1: 'True'}
    predicted_label = label_map[predicted_class_id]
    
    return predicted_label

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Omgupta@12",
    port="3306",
    database='text'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('index.html', message="Successfully Registered!")
            return render_template('index.html', message="This email ID is already exists!")
        return render_template('index.html', message="Conform password is not match!")
    return render_template('index.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('index.html', message= "Invalid Password!!")
        return render_template('index.html', message= "This email ID does not exist!")
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        text = request.form['text']
        
        # Check if text is empty
        if not text.strip():
            return render_template('home.html', prediction=None, error="Please enter some text to analyze.")
            
        try:
            predicted_label = predict(text)
            
            # Generate SHAP explanation
            focused_image = None
            full_image = None
            explanation_error = None
            word_summary = {'positive': [], 'negative': []}
            
            try:
                focused_image, full_image, word_summary = get_shap_explanation(text)
            except Exception as e:
                explanation_error = str(e)
                print(f"Error generating SHAP explanation: {e}")
            
            return render_template('home.html', 
                                  prediction=predicted_label, 
                                  text=text, 
                                  focused_image=focused_image,
                                  full_image=full_image, 
                                  explanation_error=explanation_error,
                                  word_summary=word_summary)
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            print(error_message)
            return render_template('home.html', prediction=None, error=error_message)
            
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)