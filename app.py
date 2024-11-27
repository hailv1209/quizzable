from flask import Flask, request, jsonify, render_template, redirect, url_for
import spacy
import random
import pdfplumber
from collections import Counter
import os
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('brown')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('omw-1.4')


nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback

from flashtext import KeywordProcessor


import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)


import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_pdf_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary


def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]

      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0:
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors




def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


def get_keywords(originaltext,summarytext,num_questions):
  keywords = get_nouns_multipartite(originaltext)
  print ("keywords unsummarized: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
  print ("keywords_found in summarized: ",keywords_found)

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords[:num_questions]



def generate_fill_blank(text, num_questions=20):
    
    if text is None:
        return []
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15 and not any(char.isdigit() for char in sent.text.strip())]
    
    generated_questions = set()
    mcqs = []
    
    while len(mcqs) < num_questions:
        sentence = random.choice(sentences)
        
        if len(sentence) > 200:
            continue
        
        sent_doc = nlp(sentence)
        # nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        nouns = [token.text for token in sent_doc if token.pos_ in ["NOUN", "PROPN"]]

        
        if len(nouns) < 1:
            continue
        
        subject = random.choice(nouns)
        question_stem = sentence.replace(subject, "_______", 1)
        
        if (question_stem, subject) in generated_questions:
            continue
        
        answer_choices = [subject]
        
        synonyms = get_synonyms(subject)
        similar_words = [token.text for token in nlp.vocab if token.is_alpha and token.has_vector and token.is_lower and token.similarity(nlp(subject)) > 0.5][:3]
        
        distractors = list(set(synonyms + similar_words))
        distractors = [d for d in distractors if d.lower() != subject.lower()]  # Ensure different words
        
        while len(distractors) < 3:
            new_distractor = random.choice([token.text for token in nlp(text) if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() != subject.lower() and token.text.lower() not in [d.lower() for d in distractors]])
            distractors.append(new_distractor)
        
        answer_choices.extend(random.sample(distractors, 3))
        random.shuffle(answer_choices)
        
        trivial_answer = True
        for option in answer_choices:
            if len(option) > 1:
                trivial_answer = False
                break
        
        if trivial_answer:
            continue
        
        # Check for similarity among choices
        similar_choices = True
        for i in range(len(answer_choices)):
            for j in range(i + 1, len(answer_choices)):
                if answer_choices[i].lower() == answer_choices[j].lower():
                    similar_choices = False
                    break
            if not similar_choices:
                break
        
        if not similar_choices:
            continue
        
        correct_answer = chr(64 + answer_choices.index(subject) + 1)
        mcqs.append((question_stem, answer_choices, correct_answer))
        generated_questions.add((question_stem, subject))
    
    return mcqs 


def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question


question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)


def generate_question(context,num_questions):
  generated_questions = set()
  mcqs = []
  summary_text = summarizer(context,summary_model,summary_tokenizer)
  # np = getnounphrases(summary_text,sentence_transformer_model,3)
  np =  get_keywords(context,summary_text,num_questions)
  output=""
  for answer in np:
    ques = get_question(summary_text,answer,question_model,question_tokenizer)
    distractors = get_distractors_wordnet(answer)
   
    # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
    output = output + "<b style='color:blue;'>" + ques + "</b>"
    output = output + "<br>"
    output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"+"<br>"
    if len(distractors)>0:
      for distractor in distractors[:4]:
        output = output + "<b style='color:brown;'>" + distractor+  "</b>"+"<br>"
    output = output + "<br>"
    
    # Calculate the correct answer (A, B, C, D, etc.)
    distractors = distractors[:4]
    distractors.append(answer) 
    # # Optionally shuffle the distractors if needed
    random.shuffle(distractors)

    
    correct_answer = chr(64 + distractors.index(answer) + 1)  # Use index to assign letter (A, B, C...)
        
    # Append the MCQ (question, distractors, correct answer)
    mcqs.append((ques, distractors[:4], correct_answer))
    generated_questions.add(ques)

  return mcqs




def process_questions(text, question_type, num_questions):
    if question_type == 'fill_in_blank':
        # Logic for handling fill-in-the-blank questions
        return generate_fill_blank(text,num_questions)
    else:
        # Logic for generating other types of questions mcq
        return generate_question(text,num_questions)










@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/how')
def howto():
    return render_template('howto.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # Retrieve the number of questions from the form
        num_questions = int(request.form.get('num_questions', 5))  # Default to 5 if not provided
        question_type = request.form.get('question_type','fill_in_blank')
        # Redirect to the questions route with file path and number of questions
        return redirect(url_for('questions', file_path=file_path, num_questions=num_questions,question_type=question_type))
    return redirect(request.url)

@app.route('/questions')
def questions():
    file_path = request.args.get('file_path')
    num_questions = int(request.args.get('num_questions', 5))  # Default to 5 if not provided
    question_type = request.args.get('question_type','fill_in_blank')
    text = extract_pdf_text(file_path)
    mcqs = process_questions(text, question_type, num_questions)
    
    mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
    return render_template('questions.html', mcqs=mcqs_with_index, enumerate=enumerate, chr=chr)


if __name__ == '__main__':
    app.run(debug=True)