# -*- coding: utf-8 -*-
# Optical Character Recognition (OCR) - Hebrew Resources
# Reference: https://github.com/NNLP-IL/Hebrew-Resources#optical-character-recognition-ocr

# ====================================================================================
# USE LLM TO IMPROVE TEXT - DICTA MODELS
# Reference: https://huggingface.co/dicta-il/dictabert-heq
# ------------------------------------------------------------------------------------

# from transformers import AutoModelForMaskedLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
# model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
# model.eval()
#
# sentence = 'בשנת 1948 השלים אפרים קישון את [MASK] בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
# output = model(tokenizer.encode(sentence, return_tensors='pt'))
# # the [MASK] is the 7th token (including [CLS])
# import torch
# top_2 = torch.topk(output.logits[0, 7, :], 2)[1]
# print('\n'.join(tokenizer.convert_ids_to_tokens(top_2))) # should print מחקרו / התמחותו

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pipeline use:
# from transformers import pipeline
# oracle = pipeline('question-answering', model='dicta-il/dictabert-heq')
# context = 'בניית פרופילים של משתמשים נחשבת על ידי רבים כאיום פוטנציאלי על הפרטיות. מסיבה זו הגבילו חלק מהמדינות באמצעות חקיקה את המידע שניתן להשיג באמצעות עוגיות ואת אופן השימוש בעוגיות. ארצות הברית, למשל, קבעה חוקים נוקשים בכל הנוגע ליצירת עוגיות חדשות. חוקים אלו, אשר נקבעו בשנת 2000, נקבעו לאחר שנחשף כי המשרד ליישום המדיניות של הממשל האמריקאי נגד השימוש בסמים (ONDCP) בבית הלבן השתמש בעוגיות כדי לעקוב אחרי משתמשים שצפו בפרסומות נגד השימוש בסמים במטרה לבדוק האם משתמשים אלו נכנסו לאתרים התומכים בשימוש בסמים. דניאל בראנט, פעיל הדוגל בפרטיות המשתמשים באינטרנט, חשף כי ה-CIA שלח עוגיות קבועות למחשבי אזרחים במשך עשר שנים. ב-25 בדצמבר 2005 גילה בראנט כי הסוכנות לביטחון לאומי (ה-NSA) השאירה שתי עוגיות קבועות במחשבי מבקרים בגלל שדרוג תוכנה. לאחר שהנושא פורסם, הם ביטלו מיד את השימוש בהן.'
# question = 'כיצד הוגבל המידע שניתן להשיג באמצעות העוגיות?'
# oracle(question=question, context=context)

# ====================================================================================
# END OF DICTA MODELS SECTION
# ------------------------------------------------------------------------------------

# ====================================================================================
# USE LLM TO IMPROVE TEXT - HEBREW GEMMA 11B
# Reference: https://huggingface.co/yam-peleg/Hebrew-Gemma-11B-Instruct
# ------------------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Hebrew-Gemma-11B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

chat = [
    { "role": "user", "content": "Your custom prompt here in Hebrew or English" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ====================================================================================
# END OF HEBREW GEMMA 11B SECTION
# ------------------------------------------------------------------------------------

# ====================================================================================
# USE CONFUSION MATRIX AND MARKOV CHAINS TO IMPROVE TEXT
# ------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from main import get_data


class MarkovChainPostOCR:
    """Class for processing OCR data using Markov Chain techniques.

     Attributes:
         hebrew_alphabet (str): A string of Hebrew alphabet characters.
         numbers (str): A string of number characters.
         symbols (str): A string of symbol characters.
         df (pandas.DataFrame): DataFrame to store and process OCR data.
         confusion_matrix (pandas.DataFrame): DataFrame to store the confusion matrix.
     """
    def __init__(self, get_data=get_data, hebrew_alphabet='אבגדהוזחטיכלמנסעפצקרשת', numbers='0123456789', symbols=', '):
        """Initializes the MarkovChainPostOCR with given parameters.

            Args:
                get_data: Function to get the initial data.
                hebrew_alphabet (str): Hebrew alphabet characters.
                numbers (str): Number characters.
                symbols (str): Symbol characters.
        """
        self.hebrew_alphabet = hebrew_alphabet
        self.numbers = numbers
        self.symbols = symbols
        self.df = get_data()  # Load data on initialization
        self.confusion_matrix = self.initialize_confusion_matrix()

    def preprocess_data(self):
        """Processes the initial data self.df to prepare for analysis."""
        self.df = self.df.rename(columns={'author_folder_name': 'GT',
                                          'ocr_written_on': 'OCR_text',
                                          'ocr_all_text_preprocess': 'OCR_to_fix'})
        self.df = self.df[["GT", "OCR_text", "OCR_to_fix"]]
        self.df = self.df[(self.df['GT'] != "") & (self.df['OCR_text'] != "")]

    def calculate_levenshtein_distance(self):
        """Calculates Levenshtein distance for each row in the DataFrame."""
        def calculate_levenshtein(row):
            return Levenshtein.distance(row['GT'], row['OCR_text'])

        self.df['levenshtein_distance'] = self.df.apply(calculate_levenshtein, axis=1)

    def filter_data_on_conditions(self):
        """Filters the DataFrame based on predefined levenshtein distance conditions."""
        self.df = self.df[self.df['levenshtein_distance'] < 6]
        self.df = self.df[self.df['GT'].str.len() == self.df['OCR_text'].str.len()]
        self.df = self.df[self.df['levenshtein_distance'] < 6]

    def initialize_confusion_matrix(self):
        """Initializes the confusion matrix with zeros.

               Returns:
                   pandas.DataFrame: A DataFrame initialized with zeros.
       """
        all_chars = self.hebrew_alphabet + self.numbers + self.symbols
        return pd.DataFrame(np.zeros((len(all_chars), len(all_chars))), index=list(all_chars),
                            columns=list(all_chars))

    def update_confusion_matrix(self):
        """Updates the confusion matrix based on the DataFrame content.

             Returns:
                 pandas.DataFrame: The updated confusion matrix.
        """
        matrix = self.initialize_confusion_matrix()
        for index, row in self.df.iterrows():
            for g_char, o_char in zip(row['GT'], row['OCR_text']):
                if g_char in matrix.index and o_char in matrix.columns:
                    matrix.loc[g_char, o_char] += 1
        return matrix

    def normalize_and_clean_matrix(self, matrix):
        """Normalizes and cleans the confusion matrix.

              Args:
                  matrix (pandas.DataFrame): The confusion matrix to normalize and clean.

              Returns:
                  pandas.DataFrame: The normalized and cleaned confusion matrix.
        """
        matrix_normalized = matrix.div(matrix.sum(axis=1), axis=0)
        matrix_normalized.fillna(0, inplace=True)

        matrix_no_zeros = matrix_normalized.loc[~(matrix_normalized == 0).all(axis=1)]

        diagonal_values = np.diagonal(matrix_no_zeros.values)
        rows_to_drop = matrix_no_zeros.index[diagonal_values > 0.8]
        matrix_filtered = matrix_no_zeros.drop(rows_to_drop)

        return matrix_filtered

    def plot_heatmap(self, matrix, title):
        """Plots a heatmap based on the given matrix.

          Args:
              matrix (pandas.DataFrame): The matrix to plot.
              title (str): Title for the heatmap.
        """
        plt.figure(figsize=(15, 15))
        sns.heatmap(matrix, annot=False, cmap='viridis')
        plt.title(title)
        plt.xlabel('Predicted Characters')
        plt.ylabel('Actual Characters')
        plt.show()

    def apply_corrections_to_dataframe(self, df, matrix):
        """Applies corrections to the DataFrame based on the given matrix.

             Args:
                 df (pandas.DataFrame): The DataFrame to apply corrections on.
                 matrix (pandas.DataFrame): The matrix to use for corrections.

             Returns:
                 pandas.DataFrame: The DataFrame with applied corrections.
        """
        df['Corrected_OCR'] = df.apply(lambda row: self.apply_corrections(row, matrix), axis=1)
        return df

    def apply_corrections(self, row, matrix, max_iterations=10):
        """Applies corrections to a single row based on the given matrix.

           Args:
               row (pandas.Series): A row from the DataFrame.
               matrix (pandas.DataFrame): The matrix to use for corrections.
               max_iterations (int): Maximum number of iterations for corrections.

           Returns:
               str: The corrected OCR text.
        """
        ocr_text = row['OCR_to_fix']
        for _ in range(max_iterations):
            corrected = ""
            changes_made = False

            for char in ocr_text:
                if char in matrix.columns:
                    likely_char = matrix[char].idxmax()
                    if likely_char != char:
                        corrected += likely_char
                        changes_made = True
                    else:
                        corrected += char
                else:
                    corrected += char

            ocr_text = corrected
            if not changes_made:
                break

        return corrected

    def run_example(self):
        """Runs the full example process using class methods."""
        self.preprocess_data()
        self.calculate_levenshtein_distance()
        self.filter_data_on_conditions()

        updated_matrix = self.update_confusion_matrix()

        self.plot_heatmap(updated_matrix, 'Initial Confusion Matrix Heatmap')

        matrix_filtered = self.normalize_and_clean_matrix(updated_matrix)
        self.plot_heatmap(matrix_filtered, 'Filtered Confusion Matrix Heatmap')

        df_corrected = self.apply_corrections_to_dataframe(self.df, matrix_filtered)
        print(df_corrected.head())

# example for MarkovChainPostOCR
# ocr_post_processor = MarkovChainPostOCR(get_data)
# ocr_post_processor.run_example()


# ====================================================================================
# END OF # USE CONFUSION MATRIX AND MARKOV CHAINS
# ------------------------------------------------------------------------------------

