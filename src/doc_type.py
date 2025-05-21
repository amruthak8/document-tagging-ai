import vertexai
from vertexai.generative_models import GenerativeModel, Image
import pandas as pd
import os
import fitz
from PIL import Image as Img
from src.gcs_utills import *
from dataclasses import dataclass, field
import json

from config import excel_name

import logging

@dataclass
class documenttag:
    combined_text: str = None
    final_df: pd.DataFrame = None

'''
class documenttag_segment:
    final_df: pd.DataFrame = None
    
    def __init__(self,path):
        self.path=path
        

    def text_extraction(self, state: documenttag) -> documenttag:
        print("printing in text_extraction")
        """
        Extracts and summarizes text from a PDF document using OCR via Gemini.

        This function:
        - Converts a PDF file into individual images (one per page).
        - Extracts structured text from each image using Gemini OCR.
        - Combines and formats the extracted information into a summary.

        Args:
            path (str): File path to the input PDF document.

        Returns:
            str: A formatted summary containing the document name and extracted content.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If there is an issue in image conversion or text extraction.
        """
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"PDF file not found: {self.path}")
            state.combined_text = extract_all_from_folder(self.path)
#             document_name, _ = os.path.splitext(os.path.basename(self.path))
#             image_paths = pdf_to_images(self.path)

#             extracted_sections = ''

#             for image_path in image_paths:
#                 text = extract_text_from_image_with_gemini(image_path)
#                 extracted_sections+= text.strip()

#             state.combined_text = f"Name of the document: {document_name}, and it contains information: {state.combined_text}"
            return state

        except FileNotFoundError as fnf:
            raise fnf
        except Exception as e:
            raise Exception(f"Error during text extraction: {type(e).__name__} - {e}")
'''

class documenttag_segment:
    final_df: pd.DataFrame = None
    
    def __init__(self, path):
        self.original_path = path
        self.local_path = self._prepare_local_path(path)
        self._should_cleanup = self.local_path.startswith(tempfile.gettempdir())

    def _prepare_local_path(self, path: str) -> str:
        if path.startswith("gs://"):
            print(f"Downloading files from GCS: {path}")
            
            logging.info(f"Downloading files from GCS: {path}")
            return self._download_gcs_folder(path)
        elif os.path.exists(path):
            return path
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def _download_gcs_folder(self, gcs_uri: str) -> str:
        bucket_name, prefix = gcs_uri[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        tmp_dir = tempfile.mkdtemp()
        blobs = list(client.list_blobs(bucket, prefix=prefix))

        if not blobs:
            raise FileNotFoundError(f"No files found at {gcs_uri}")

        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            rel_path = os.path.relpath(blob.name, prefix)
            local_file_path = os.path.join(tmp_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)

        print(f"Downloaded GCS contents to: {tmp_dir}")
        
        logging.info(f"Downloaded GCS contents to: {tmp_dir}")
        return tmp_dir

    def text_extraction(self, state: documenttag) -> documenttag:
        print("Running text_extraction")
        
        logging.info("Running text_extraction")
        try:
            # Your existing logic that takes a local folder path
            state.combined_text = extract_all_from_folder(self.local_path)
            
            print("state.combined_text: ", state.combined_text)
            
            logging.info("state.combined_text: ", state.combined_text)
            
            logging.info("End of text_extraction")
            
            return state

        except Exception as e:
            raise Exception(f"Error during text extraction: {type(e).__name__} - {e}")
    
    def cleanup(self):
        """Clean up temp folder if created."""
        if self._should_cleanup and os.path.exists(self.local_path):
            print(f"Cleaning up temp folder: {self.local_path}")
            
            logging.info(f"Cleaning up temp folder: {self.local_path}")
            shutil.rmtree(self.local_path, ignore_errors=True)
            
    def get_audience(self, file_name):
        
        # Path to your Excel file
        excel_path = excel_name # <-- Replace with actual path
        document_name = file_name # <-- Replace with the document name you're searching for
        
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Check if the document exists
        if document_name in df['NAME'].values:
            audience = df.loc[df['NAME'] == document_name, 'Audience'].values[0]
            #print(f"Audience for '{document_name}': {audience}")
            return audience
        else:
            #print(f"Document '{document_name}' not found.")
            return ""
    
    def get_doc_type(self, combined_text, audience):
        
        print("printing in get_doc_type")
        
        logging.info("printing in get_doc_type")
        """
        Analyze input content and identify the type of the document using Gemini.

        Args:
            content (str): The input text to identify the type of the document.

        Returns:
            str: The identified document type, or "No industry segment detected".
        
        Raises:
            Exception: If the Gemini model fails to process the input.
        """
        system_prompt= """You are a highly specialized language model designed to identify and classify document types in text. 
    Your task is to:
    - Determine whether the given input contains any contextual reference to a document type.
    - Identify and classify the content into one relevant document type based on the following definitions:
    
    
    Document Types:
    Litho: Lithos are short, customer-facing product resources such as sell sheets and spec sheets. These documents are typically 1–3 pages in length and are designed to provide concise product information. Litho documents are short, visually structured, and customer- or client-facing marketing or product information sheets. They typically:
    - Emphasize ready-to-eat or ready-to-use food products.
    - Emphasize convenience, versatility, and ease of use.
    - Use a concise, benefit-oriented tone with bullet points or callouts.
    - Include product codes, packaging information, nutrition facts, and other commercial specs.
    - Do not contain cooking instructions or ingredients for preparing a dish from scratch.
    
    Selling Guide: Selling Guides are longer customer-facing documents (usually 4+ pages) that support sales conversations by highlighting product benefits and key selling points.
    
    Recipe: This type includes recipe books and culinary idea documents. It focuses on food preparation instructions and cooking inspiration. Recipe documents are instructional and focus on how to prepare a dish or food item. They typically:
    - Include a list of ingredients and step-by-step preparation instructions.
    - Are centered around culinary creation, food inspiration, or menu development.
    - May be part of a recipe book, culinary concept document, or meal idea sheet.
    - May explore food presentation, flavor pairings, and preparation techniques.
    - Are not focused on promoting prepackaged or commercial food products.
    
    Playbook: Playbooks provide structured guidance on how to navigate specific situations or scenarios. These documents are often used internally for strategic or procedural purposes.
    
    Rebate: Rebate documents outline promotional rebate programs, detailing eligible products, savings offers, and often include submission forms for claiming rebates.
    
    Trade: Trade documents relate to the trade process and typically include training materials, release information, or guidelines specific to trade activities.
    
    Report: Reports are data-driven documents, often formatted in tables or charts. They emphasize metrics, performance data, or analytical findings.
    
    Video: This category includes multimedia content, particularly videos (commonly hosted on platforms like YouTube), which may explain, promote, or demonstrate content.
    
    Overview: A broad category encompassing general summary documents. Overview materials provide high-level information across a wide range of subjects.
    
    Training Material: These are internal resources designed for instructional purposes, including guides, presentations, and e-learning content aimed at educating staff or partners.
    
    Other: A catch-all category for any document type that does not clearly fall into the categories above.
    
    When analyzing a piece of text, always:
    - Detect if the text is or is not relevant to any one of the above document types.
    - Focus on the contextual relevance of the input content to each document type’s definition — not just the presence of specific keywords. Carefully consider the purpose, format, and audience implied by the text. 
    
    Return only the single most relevant document type.
    - If a relevant type is found, state it clearly and briefly explain its relevance.
    - If no relevant type is found, respond with: "No document type detected."

 """
    
        user_prompt=f"""
        Input Text: {combined_text}
    Determine if the above text contains clear references to any document types listed below. Match only if the text directly aligns with a definition. Quote supporting text if applicable. If no clear match exists, respond: "No document type detected".
    
    The audience type is {audience}.
    
    Instructions:
    - If the audience is Internal, then the document type cannot be 'Litho'.
    - Please make sure to return only one most relevant document type output.
    
        Output Format (in JSON):
                    [
                      {{
                        "document name": "Name of Document",
                        "match": [
                          {{
                            "doc_type": "Name of the document type",
                            "justification": "Quoted supporting text from input"
                          }}
                        ]
                      }}
                    ]

                    If no matches are found, return the same structure with an empty 'match' array:
                    [
                      {{
                        "document name": "Name of Document",
                        "match": [
                        {{
                            "doc_type": "Name of the document type",
                            "justification": "Quoted supporting text from input"
                          }}
                          ]
                      }}
                    ]
                    """

        try:
            model = GenerativeModel("gemini-2.0-flash-001", system_instruction=system_prompt)
            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 0.1,
                "top_p": 0.4,
            }

            response = model.generate_content(
                [user_prompt],
                generation_config=generation_config
            )
            response=response.text
            response = response.replace("```json", "")
            response = response.replace("```", "")
            doc_type = json.loads(response)
            
            logging.info("End of get_doc_type")
            
            return doc_type

        except Exception as e:
            raise Exception(f"Error during doc type analysis: {type(e).__name__} - {e}")

    
    def resulting_doc_type(self, state: documenttag) -> documenttag:
        
        print("printing in resulting_doc_type")
        
        logging.info("printing in resulting_doc_type")
        result = []
        for index, row in state.combined_text.iterrows():
            file_name = row[0]  # or row['file'] if you renamed columns
            file_content = row[1]
            page_doc_type = row[2]
            if page_doc_type == "":
                audience = self.get_audience(file_name)
                # doc_type_gt = self.get_doc_type_gt(file_name)
                segment = self.get_doc_type(file_content, audience)
                print("***************Doc Type*******************")
                
                logging.info("***************Doc Type*******************")
                dataframe, page_doc_type = extract_segment_data(segment)
                
            else:
                rows = []
                rows.append({
                    "document name": row[0],
                    "doc_type": page_doc_type
                    # "gt_doc_type": doc_type_gt
                })
                dataframe = pd.DataFrame(rows)
                
            result.append(dataframe)
        
        final_df = pd.concat(result, ignore_index=True)
        final_df.to_csv("doc_type_results_104_with_gt.csv", index=False)
        
        logging.info("End of resulting_doc_type")
        return state

            
            


