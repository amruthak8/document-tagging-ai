import concurrent.futures
from dataclasses import dataclass, field
import vertexai
from vertexai.generative_models import GenerativeModel,Image
import pandas as pd
import os
import fitz
from PIL import Image as Img
import json
from google.cloud import bigquery
from datetime import datetime
from typing import Optional, List
import warnings
import time
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import traceback
import re

import logging

from src.gcs_utils import GCSFileHandler
import config

gcs=GCSFileHandler(config.BUCKET_NAME)

# Ignore warnings
warnings.filterwarnings("ignore")

vertexai.init(
        project="aia-aiva-poc-dna-dev-c83dcb",  
        location="us-central1",
        api_endpoint="us-central1-aiplatform.googleapis.com"
    )


@dataclass
class documenttag:
    aggregate: Annotated[list, operator.add]
    combined_text: str = None
    text_df: pd.DataFrame = None
    industry_df: pd.DataFrame = None
    bpc_df: pd.DataFrame = None
    account_focus_df: pd.DataFrame = None
    doc_type_df: pd.DataFrame = None
    Industry_segment: Optional[List[str]] = None 
    bpc_code: Optional[List[str]] = None 
    final_df: pd.DataFrame = None
    
    
class documenttag_segment():
    final_df: pd.DataFrame = None
    # aggregate: Annotated[list, operator.add]
    
    def __init__(self,path):
        self.path=path
        
    
    def extract_data(self,state: documenttag):
        
        """
        Extracts text data from all documents in the specified input folder.

        This method retrieves and processes all document files located in the folder 
        defined by `self.path`, using the `gcs.extract_all_from_folder` utility.

        Args:
            state (documenttag): The current state of the document tagging process.
                                 (Note: This argument is required by LangGraph structure but is not used directly here.)

        Returns:
            dict: A dictionary containing a DataFrame under the key `"text_df"`, 
                  which holds the extracted text data from the documents.

        Example:
            output = self.extract_data(state)
            df = output["text_df"]
        """
        
        text_df = gcs.extract_all_from_folder(self.path)
        
        return {"text_df": text_df}
    
    def get_audience(self, file_name):
        
        """
        Retrieves the target audience associated with a given document name from an Excel file.

        This method reads an Excel sheet (configured via `config.excel_name`) and looks up 
        the `Audience` value corresponding to the provided document name in the `NAME` column.

        Args:
            file_name (str): The name of the document whose audience needs to be fetched.

        Returns:
            str: The target audience if the document name exists in the Excel file; 
                 otherwise, an empty string.

        Example:
            audience = self.get_audience("example_doc.pdf")
            if audience:
                print(f"Target audience: {audience}")
            else:
                print("Document not found.")
        """
        
        # Path to your Excel file
        excel_path = config.excel_name # <-- Replace with actual path
        document_name = file_name # <-- Replace with the document name you're searching for
        
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Check if the document exists
        if document_name in df['NAME'].values:
            audience = df.loc[df['NAME'] == document_name, 'Audience'].values[0]
            
            return audience
        else:
            
            return ""
    
    
    #####################BASE PRODUCT TAGGING#################
    def base_product_codes(self,content):
        
        """
        Extracts valid Base Product Codes (BPCs) from the given text content using an LLM.

        This method uses the Gemini model to identify and extract valid numeric Base Product Codes 
        based on strict rules. It ensures only explicitly labeled BPCs of at least 9 digits are returned.

        Args:
            content (str): The raw textual content from which BPCs need to be extracted.

        Returns:
            list[dict]: A list containing a dictionary with:
                - "document name": A placeholder or inferred name of the document.
                - "base product codes": A list of valid 9+ digit BPCs as strings.

            Example:
                [
                    {
                        "document name": "Example_Document.pdf",
                        "base product codes": ["106249000", "106252000"]
                    }
                ]
        
        Raises:
            None explicitly. Returns a default response if model fails or output is invalid.
        """
        
        system_prompt = """
    You are a specialized data extraction assistant. Your task is to extract only valid **Base Product Codes (BPCs)** from the given input. These are numeric codes found exclusively in the **'BPC' column** or fields explicitly labeled as **Base Product Code**, **BPC**, or **Base Product**.

    ### STRICT Extraction Rules:

    - A valid BPC is a **numeric code with at least 9 digits** (e.g., 106249000).
    - Only extract codes that appear:
      - In a **'BPC' column** of a table or product list
      - In clearly labeled fields (e.g., “Base Product Code: 106249000”)
    - Do **NOT** extract codes that:
      - Are fewer than 8 digits
      - Appear in **product names**, descriptions, or ingredient listings (e.g., “White Chunk Raspberry Scone (08151)”,  "BONICI® Parbaked Raised Thin Extra Crispy Pizza Crust 2100224000"
    )
      - Are found inside **parentheses**
      - Are GTINs, Pack Sizes, or values from other columns

    - Do not infer or guess any codes. Only extract values that are explicitly associated with the **'BPC'** label or field.
    - Make sure the extracted codes not less than 8 digits
    ### Output Format:

    Return a JSON object in the following format:

    [
      {
        "document name": "Name of Document",
        "base product codes": ["106249000", "106252000", "106236000", "106237000"]
      }
    ]

    If no valid BPC is found, return:

    [
      {
        "document name": "Name of Document",
        "base product codes": []
      }
    ]
    """

        user_prompt = f"""Input Text: {content}\n\nTask: Extract all numeric Base Product Codes (BPC) following the rules above."""
        
        try:
            model = GenerativeModel("gemini-2.0-flash-001", system_instruction=system_prompt)

            generation_config = {
                "max_output_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.4,
            }

            response = model.generate_content(
                [user_prompt],
                generation_config=generation_config
            )


            bpc_string = response.text.strip()
            bpc_string = bpc_string.replace("```", "").replace("json", "").strip()
            bpc_string= json.loads(bpc_string)
            # print(bpc_string)

            return bpc_string
        
        except Exception as e:
            logging.exception("Unexpected error during BPC extraction:")
    
    def Universal_product_code(self,content):
        
        """
        Extracts valid Universal Product Codes (UPCs) from the provided text using an LLM.
        
        This method leverages the Gemini model to extract UPC or GTIN codes based on clearly defined
        and strict criteria. It only extracts 12–14 digit numeric codes from labeled fields such as
        "UPC", "GTIN", etc., and normalizes them by removing hyphens, spaces, and non-numeric characters.
        
        Args:
            content (str): The raw input text content from which UPC or GTIN codes need to be extracted.
        
        Returns:
            list[dict]: A list containing a single dictionary with:
                - "document name": A placeholder or model-inferred name of the document.
                - "upc codes": A list of valid 12–14 digit UPC strings extracted from the input.

            Example:
                [
                  {
                    "document name": "Example_Document.pdf",
                    "upc codes": ["10094562062368", "10094562062375"]
                  }
                ]
        
        Notes:
            - If both UPC and GTIN are found, only UPCs are retained.
            - GTINs are considered only if no UPC is available.
            - Extraction is strictly limited to explicitly labeled fields (e.g., "UPC Code", "GTIN").
            - Codes in product names, descriptions, parentheses, or unlabeled fields are ignored.
            - Codes are normalized to remove hyphens and whitespace.

        Raises:
            json.JSONDecodeError: If the model returns invalid JSON.
            Exception: Other unexpected issues can arise from model response or formatting.
        """
        
        system_prompt = """
        You are a specialized data extraction assistant. Your task is to extract valid **UPC (Universal Product Codes)** from the input text using strict rules.

        ---

        ### ✅ VALID CODE DEFINITION:

        - A **valid UPC** is a **12 to 14-digit numeric code**, possibly separated by hyphens or spaces.
        - A **GTIN** (Global Trade Item Number) is a broader term that can also be 12–14 digits and may be interchangeable with UPC in certain fields.

        Treat **GTIN and UPC as equivalent**, but **prioritize UPC** when both are available:
        - If a UPC is present, extract the UPC **only**.
        - If no UPC is found, and a GTIN is present in a valid field, extract the GTIN as a UPC.

        ---

        ### ✅ VALID FIELDS FOR EXTRACTION:

        Only extract codes from fields explicitly labeled as:

        - "UPC"
        - "UPC Code"
        - "UPC Case Code"
        - "Universal Product Code"
        - "GTIN"
        - "GTIN Code"
        - "GTIN Case Code"

        Codes must **only** be extracted from these fields.

        ---

        ### 🔧 NORMALIZATION RULES:

        - Remove any hyphens or spaces.
        - Only return pure 12–14 digit numeric codes.
        - Discard any non-numeric characters.

        ---

        ### ❌ DO NOT Extract Codes That:

        - Are in parentheses — e.g., “(123456789012)” is invalid.
        - Appear in product names, descriptions, marketing copy, or ingredients.
        - Appear in unrelated fields not clearly labeled as UPC/GTIN.
        - Are fewer than 12 digits or more than 14 digits after normalization.
        - Are inferred — extract only if explicitly stated in a valid field.
        - Make sure the extracted codes not less than 8 digits

        ---

        ### ⚠️ EXAMPLES TO IGNORE:

        - Pillsbury™ Croissant Dough 1.25 oz (13444) ← Invalid (in parentheses)
        - Pillsbury™ Biscuit (08151) ← Not a UPC

        ---

        ### 📦 Output Format (strict JSON):

        Return the output in this format:

        [
          {
            "document name": "Name of Document",
            "upc codes": ["10094562062368", "10094562062375", "10094562324992"]
          }
        ]

        If no valid UPCs or GTINs are found:

        [
          {
            "document name": "Name of Document",
            "upc codes": []
          }
        ]
        """


        user_prompt = f"""Input Text: {content}\n\nTask: Extract all valid UPC codes according to the instructions above."""
        
        try:
            model = GenerativeModel("gemini-2.0-flash-001", system_instruction=system_prompt)

            generation_config = {
                "max_output_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.4,
            }

            response = model.generate_content(
                [user_prompt],
                generation_config=generation_config
            )

            upc_string = response.text.strip()
            upc_string = upc_string.replace("```", "").replace("json", "").strip()
            upc_string= json.loads(upc_string)

            return upc_string
        
        except Exception as e:
            logging.exception("Unexpected error during Universal_product_code:")


    def query_distinct_products(self, upc_codes: list) -> tuple[bigquery.table.RowIterator, list] | None:
        """
        Queries BigQuery to retrieve distinct product details for given UPC codes.

        Parameters:
            upc_codes (list): List of UPC codes to query.

        Returns:
            Tuple of:
                - bigquery.table.RowIterator: Iterator over query results.
                - List of base product codes.
            Returns None if UPC list is empty or query fails.
        """

        if not upc_codes:
            # print("No UPC codes provided. Skipping query.")
            
            logging.info("No UPC codes provided. Skipping query.")
            return None

        project_id = "aia-aiva-poc-dna-dev-c83dcb"
        dataset_id = "transient"

        client = bigquery.Client(project=project_id)

        # Convert list of UPC codes to comma-separated quoted strings
        upc_list = ', '.join([f"'{code}'" for code in upc_codes])

        query = f"""
            SELECT DISTINCT 
            ean_upc_fully_qualified_cd, gtin,
            LTRIM(base_product_cd, '0') AS base_product_cd, 
            base_product_desc 
        FROM `edw-prd-e567f9.cnf.dim_naf_product`
        WHERE LTRIM(ean_upc_fully_qualified_cd,'0') IN ({upc_list}) 
        OR gtin IN ({upc_list})
        """

        base_product_code = []

        try:
            query_job = client.query(query)
            results = query_job.result()  # Wait for job to complete
            
            for row in results:
                base_product_code.append(row['base_product_cd'])
                
            return results, base_product_code
        
        except Exception as e:
            logging.exception("Error occurred in query_distinct_products:", e)
            return None

        
    def replace_upc_with_bpc_in_order(self,data, bpc_list):
        """
        Replaces UPC codes with BPC values in order, renaming the field to 'bpc codes'.

        Parameters:
        - data (list of dict): Original data with 'upc codes'.
        - bpc_list (list of str): BPC values in the same order as the UPCs.

        Returns:
        - list of dict: Updated structure with BPCs replacing UPCs, and key renamed to 'bpc codes'.
        """
        
        try:
            updated_data = []

            for item in data:
                upc_codes = item.get('upc codes', [])

                # Replace UPCs with BPCs based on order
                updated_item = item.copy()
                updated_item.pop('upc codes', None)  # Remove old key
                updated_item['base product codes'] = bpc_list.copy()  # Add new key with BPCs
                updated_data.append(updated_item)

            return updated_data
        
        except Exception as e:
            logging.exception("Error occurred in replace_upc_with_bpc_in_order:", e)
            return []
    
    def extract_base_product_code(self,data, bpc_list):
        """
        Replaces UPC codes with BPC values in order, renaming the field to 'bpc codes'.

        Parameters:
        - data (list of dict): Original data with 'upc codes'.
        - bpc_list (list of str): BPC values in the same order as the UPCs.

        Returns:
        - list of dict: Updated structure with BPCs replacing UPCs, and key renamed to 'bpc codes'.
        """
        
        try:
            updated_data = []

            for item in data:
                upc_codes = item.get('upc codes', [])

                # Replace UPCs with BPCs based on order
                updated_item = item.copy()
                updated_item.pop('upc codes', None)  # Remove old key
                updated_item['base product codes'] = bpc_list.copy()  # Add new key with BPCs
                updated_data.append(updated_item)

            return updated_data
        
        except Exception as e:
            logging.exception("Error occurred in extract_base_product_code:", e)
            return []
    
    def extract_bpc_codes(self, bcp_json):
        """
        Extracts base product codes into a DataFrame.

        Parameters:
            bcp_json (list or dict): JSON list or single dict with 'document name' and 'base product codes'.

        Returns:
            pd.DataFrame: DataFrame with columns: 
                          'document name', 'base product codes'
        """
        
        try:
            # Ensure input is a list
            if not isinstance(bcp_json, list):
                bcp_json = [bcp_json]

            bcp_rows = []
            for entry in bcp_json:
                doc_name = entry.get("document name", "Unknown Document")
                bpcs = entry.get("base product codes", [])

                # Ensure base product codes is a list before joining
                if isinstance(bpcs, list):
                    bpc_str = ", ".join(bpcs)
                else:
                    bpc_str = str(bpcs)

                bcp_rows.append({
                    "document name": doc_name,
                    "base product codes": bpc_str
                })

            return pd.DataFrame(bcp_rows)
        
        except Exception as e:
            logging.exception("Failed to extract BPC codes into DataFrame:")
            # Return empty DataFrame with expected columns on error
            return pd.DataFrame(columns=["document name", "base product codes"])


    def base_product_code_tag(self, state: documenttag):
        
        """
        Extracts Base Product Codes (BPCs) from document content using a multi-step strategy.

        The method iterates over all documents in `state.text_df` and attempts to extract Base Product Codes (BPCs) 
        using two strategies:

        1. **Direct Extraction**: Applies a strict rule-based model to extract BPCs directly from the content.
        2. **Fallback via UPC**:
            - Extracts UPC codes using a dedicated extraction method.
            - Queries BigQuery to map UPCs to BPCs.
            - Replaces UPCs with mapped BPCs in the result structure.

        For each document, it builds a structured result and handles errors gracefully. 
        The final result is saved to a CSV file and returned as a DataFrame.

        Args:
            state (documenttag): A state object containing a DataFrame `text_df` with file names and their content.

        Returns:
            dict: A dictionary with the key `"bpc_df"` containing a pandas DataFrame with:
                - "document name"
                - "base product codes" columns.
        """
        # state.text_df = pd.read_csv("text_df_all.csv")
        result = []
        errors = []
        
        try:

            for index, row in state.text_df.iterrows():
                time.sleep(1.5)
                file_name = row[0]  # Adjust if using column names
                file_content = row[1]

                try:
                    # Step 1: Try to extract BPC directly
                    bpc_direct = self.base_product_codes(file_content)
                    has_direct_bpc = any(
                        entry.get('base product codes') for entry in bpc_direct
                    )

                    if has_direct_bpc:
                        dataframe = self.extract_bpc_codes(bpc_direct)
                        result.append(dataframe)
                        # # print(f"[{file_name}] Base Product Code found directly.")

                        logging.info(f"[{file_name}] Base Product Code found directly.")
                        continue  # Go to next file

                    # Step 2: Extract UPCs
                    upc_result = self.Universal_product_code(file_content)
                    upc_codes = upc_result[0].get("upc codes") if upc_result else []

                    if not upc_codes:
                        # # print(f"[{file_name}] No valid UPCs found. Skipping.")

                        logging.info(f"[{file_name}] No valid UPCs found. Skipping.")
                        continue

                    # Step 3: Query BPCs using UPCs
                    query_result = self.query_distinct_products(upc_codes)
                    if not query_result:
                        # # print(f"[{file_name}] No mapping found for UPCs.")

                        logging.info(f"[{file_name}] No mapping found for UPCs.")
                        continue

                    _, bpc_mapped = query_result
                    replaced_bpc_data = self.replace_upc_with_bpc_in_order(upc_result, bpc_mapped)

                    # Step 4: Extract structured BPC data
                    dataframe = self.extract_bpc_codes(replaced_bpc_data)
                    result.append(dataframe)

                except Exception as e:
                    error_message = f"[{file_name}] Error: {str(e)}"

                    traceback.print_exc()
                    errors.append({"file_name": file_name, "error": str(e)})

            final_df = pd.concat(result, ignore_index=True)
            #final_df.to_csv("bpc_results.csv", index=False)

            # Step 5: Merge results into state
            if result:
                bpc_df = pd.concat(result, ignore_index=True)
            else:
                bpc_df = pd.DataFrame(columns=["document name", "base product codes"])


            return {"bpc_df": bpc_df}
        
        except Exception as e:
            logging.exception("Unhandled error in base_product_code_tag:", e)
            return {"bpc_df": pd.DataFrame(columns=["document name", "base product codes"])}


    #####################INDUSTRY SEGMENT TAGGING#################
    
    def industry_segment(self, content):
        
        """
        Classifies input text into structured foodservice industry segments and sub-segments using Gemini 2.0 model.

        The method sends the input `content` to a Gemini 2.0 model with detailed system instructions to:
          - Detect relevant foodservice segments and all applicable sub-segments.
          - Map sub-segments hierarchically under their parent segments.
          - Provide quoted justifications for each match.
          - Avoid overclassification and weak inferences.

        If no segment can be confidently detected, the method returns a default structure indicating no match, along with justification.

        Args:
            content (str): The raw input text or document content to classify.

        Returns:
            list[dict]: A JSON-style Python list containing:
                - `"document name"`: (str) Placeholder name for the document.
                - `"matches"`: List of dictionaries, each with:
                    - `"segment"`: Matched industry segment or `"No industry segment detected"`.
                    - `"sub-segment"`: List of matched sub-segments (empty if none).
                    - `"justification"`: Quoted or explanatory text for the classification.

        Example Output:
            [
              {
                "document name": "Name of Document",
                "matches": [
                  {
                    "segment": "Healthcare",
                    "sub-segment": ["Hospitals", "Other Healthcare"],
                    "justification": "Provides meals for patients in hospitals and facilities"
                  }
                ]
              }
            ]

        Raises:
            json.JSONDecodeError: If model response is not valid JSON.
            Exception: For unexpected issues in model interaction or parsing.
        """
        system_prompt= """You are a foodservice industry segment classifier and document analyst. Your role is to examine input text and determine its relevance to foodservice industry segments based on strict yet context-aware criteria. 

### Objectives:
1. **Detect Relevance**: Determine if the input text contains direct or clearly inferred references to any defined foodservice segments.
2. **Evaluate Contextual Fit**: Analyze operations, environments, target audiences, product attributes, or service features to align with the definitions below.
3. **Consultant-Level Reasoning**:
   - Identify **explicit** matches (e.g., “hospital cafeteria” → Healthcare).
   - Infer **implicit** matches when strong context exists (e.g., “serving patients” → Healthcare; “menu variety for students” → Education).
4. **Ensure Precision**: Only tag a segment if there is a justifiable, supported match.
5. **Support Each Match with Quoted Justification**: All matches must include quoted supporting phrases from the input text.

### Industry Segment Definitions:
    - "Education: The education foodservice segment encompasses meal programs in K-12 schools, colleges, 
    and universities, focusing on providing nutritious, cost-effective meals to students, faculty, and staff.
    It plays a key role in supporting student health, academic performance, and well-being, often adhering to
    government nutrition standards. This segment balances large-scale meal production with menu variety, cultural 
    preferences, and budget constraints."

    - "Healthcare: The healthcare foodservice segment provides nutritionally balanced meals tailored to the medical
    and dietary needs of patients in hospitals, long-term care facilities, and rehabilitation centers. 
    It also supports staff and visitors with cafeteria-style dining while maintaining strict hygiene, safety, 
    and regulatory standards. This segment emphasizes therapeutic diets, patient satisfaction, and operational 
    efficiency."

    - "K-12: The foodservice segment of K–12 (kindergarten through 12th grade) is a specialized part of the 
    food industry that focuses on providing nutritious, cost-effective meals to students in public and private 
    elementary, middle, and high schools."

    - "Lodging: The lodging segment of foodservice includes dining operations within hotels, resorts, 
    and other accommodations, offering a range of services from casual room service to fine dining experiences. 
    These operations aim to enhance the guest experience through convenience, quality, and customization, 
    often featuring diverse menus to cater to travelers' tastes and expectations. Flexibility, consistency, 
    and service excellence are essential to meet both leisure and business traveler demands."

    - "Other: "Other" foodservice refers to operations that don’t fall neatly into traditional categories 
    like restaurants, schools, or healthcare. This segment includes foodservice in correctional facilities, 
    military installations, transportation centers, and corporate offices. These settings often have specific 
    audiences and operational constraints, requiring tailored menus, service models, and logistical planning."

    - "Other Food Locations: Other foodservice segments that don’t fit into clear categories are often referred 
    to as “nontraditional” or “miscellaneous” segments. These include operations in places like correctional 
    facilities, military bases, transportation hubs (airports, train stations), and corporate or government offices. 
    Though varied in nature, they all serve specific, often captive, audiences with unique service needs and 
    logistical requirements."

    - "Recreation: Recreation foodservice serves venues focused on leisure and entertainment, such as stadiums, 
    theme parks, movie theaters, and casinos. These operations emphasize speed, convenience, and crowd-pleasing menu 
    options while enhancing the overall guest experience. Balancing efficiency with fun, they often incorporate 
    themed offerings and branded concepts to align with the venue's atmosphere."

    - "Restaurants: The restaurant foodservice segment includes a wide range of establishments, from quick-service 
    chains to fine dining restaurants, all focused on preparing and serving meals to customers. This segment is highly 
    diverse and customer-driven, with a strong emphasis on menu innovation, service quality, and ambiance. 
    It plays a major role in the foodservice industry by catering to various dining preferences and social experiences."

    - "Retail Food: The retail foodservice segment refers to food and beverage offerings within retail environments 
    like grocery stores, convenience stores, and big-box retailers. It includes ready-to-eat meals, deli counters, 
    salad bars, and in-store cafés, catering to consumers looking for convenient, on-the-go dining options. 
    This segment bridges traditional retail and foodservice, emphasizing speed, accessibility, and value."

### Output Instructions:
- If segment(s) detected: return each matched segment with quoted text justification.
- If **no segment** clearly applies, respond with `"No industry segment detected"` and explain why (e.g., lacks context, too vague, not foodservice-related).

     """

        user_prompt=f"""
        Input Text:{content}
        
        Task:"Identify and classify this content against foodservice industry segments. Match only if there is strong alignment with the definitions. 
        Quote supporting text. If no clear match, return "No industry segment detected"."

        Output Format (in JSON):
                    [
                      {{
                        "document name": "Name of Document",
                        "matches": [
                          {{
                            "segment": "Name of matched segment",
                            "justification": "Quoted supporting text from input"
                          }},
                          ...
                        ]
                      }}
                    ]

                    If no matches are found, return the same structure with an empty 'matches' array:
                    [
                      {{
                        "document name": "Name of Document",
                        "matches": [
                        {{
                            "segment": "No industry segment detected",
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
                generation_config=generation_config)
            response = response.text
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response= json.loads(response)

            return response
        
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error: {str(e)}")
            logging.debug(f"Raw response: {response_text}")
            return [{
                "document name": "Name of Document",
                "matches": [{
                    "segment": "No industry segment detected",
                    "sub-segment": "",
                    "justification": "Model response could not be parsed as valid JSON."
                }]
            }]
    
        except Exception as e:
            logging.error(f"Error in industry_segment classification: {str(e)}")
            traceback.print_exc()
            return [{
                "document name": "Name of Document",
                "matches": [{
                    "segment": "No industry segment detected",
                    "sub-segment": "",
                    "justification": f"Model processing failed: {str(e)}"
                }]
            }]
        
    
    def extract_industry_segment(self,industry_json):
        """
        Extracts industry segment and sub-segment data into a DataFrame.

        Parameters:
            industry_json (list): JSON list with 'document name' and 'matches'.

        Returns:
            pd.DataFrame: DataFrame with columns: 
                          'document name', 'industry segment', 'industry sub-segment'
        """
        
        try:
            industry_rows = []
            for entry in industry_json:
                doc_name = entry.get("document name", "Unknown Document")
                matches = entry.get("matches", [])

                seen_segments = set()
                seen_subsegments = set()
                ordered_segments = []
                ordered_subsegments = []

                for match in matches:
                    segment = match.get("segment")
                    subsegments = match.get("sub-segment", [])
                    if isinstance(subsegments, str):
                        subsegments = [subsegments]

                    if segment and segment not in seen_segments:
                        seen_segments.add(segment)
                        ordered_segments.append(segment)

                    for sub in subsegments:
                        if sub and sub not in seen_subsegments:
                            seen_subsegments.add(sub)
                            ordered_subsegments.append(sub)

                industry_rows.append({
                    "document name": doc_name,
                    "industry segment": ", ".join(ordered_segments),
                    "industry sub-segment": ", ".join(ordered_subsegments)
                })

            return pd.DataFrame(industry_rows)
        
        except Exception as e:
            logging.error(f"Error in extract_industry_segment: {str(e)}")
            traceback.print_exc()
            raise  # Optional: re-raise to let the caller decide how to handle
    
    
    def industry_segment_tag(self, state: documenttag)-> documenttag:
        
        """
        Applies industry segment tagging to a batch of documents in the given state.

        For each document (row) in `state.text_df`, the method:
          1. Extracts the file name and content.
          2. Invokes the `industry_segment()` method to classify the content into foodservice segments/sub-segments.
          3. Parses the classification result into a structured DataFrame.
          4. Aggregates all results and stores them in `industry_df` for downstream use.

        Any errors encountered during processing are logged and collected.

        Args:
            state (documenttag): A state object containing `text_df`, a DataFrame with documents to classify.
                Expected columns:
                    - Column 0: Document name or ID
                    - Column 1: Document content (text)

        Returns:
            documenttag: Updated state dictionary with:
                - `industry_df` (DataFrame): Columns include `document name` and `industry segment`.
                  Saved as `industry_df.csv` locally for inspection.

        Raises:
            Logs exceptions per document but does not interrupt processing of remaining documents.
        """
        
        result = []
        errors = []

        for index, row in state.text_df.iterrows():
            time.sleep(1.5)
            file_name = row[0]  # or use column name if applicable
            file_content = row[1]

            try:
                industry_json = self.industry_segment(file_content)
                dataframe = self.extract_industry_segment(industry_json)
                result.append(dataframe)
                
                logging.info(f"[{file_name}] Industry segment extracted successfully.")
                
            except Exception as e:
                error_message = f"[{file_name}] Error: {str(e)}"
                # print(error_message)
                
                logging.info(error_message)
                traceback.print_exc()
                errors.append({"file_name": file_name, "error": str(e)})
        
        
        # Set dataframe even if empty to avoid attribute errors later
        if result:
            industry_df = pd.concat(result, ignore_index=True)
        else:
            industry_df = pd.DataFrame(columns=["document name", "industry segment"])
            
        '''
        
        final_df = pd.concat(result, ignore_index=True)
        #final_df.to_csv("industry_df.csv", index=False)
        
        '''
        '''
        # Save to CSV for inspection
        try:
            industry_df.to_csv("industry_df.csv", index=False)
            logging.info("industry_df.csv saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save industry_df.csv: {str(e)}")
            traceback.print_exc()
        '''
        return {"industry_df": industry_df}


    ###################Document Tagging#################
    def get_doc_type(self, combined_text, audience):
        
        # print("printing in get_doc_type")
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
    - Promote ready-to-eat or ready-to-use food products.
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
            return doc_type

        except Exception as e:
            raise Exception(f"Error during doc type analysis: {type(e).__name__} - {e}")
    
    def extract_doc_type_data(self, json_data):
        """
        Extracts document type data into a DataFrame.

        Parameters:
            json_data (list): JSON list with 'document name' and 'matches'.

        Returns:
            pd.DataFrame: DataFrame with columns:
                          'document name', 'doc_type'
        """
        rows = []
        for entry in json_data:
            
            try:
                doc_name = entry.get("document name", "Unknown Document")
                matches = entry.get("match", [])

                seen_doc_types = set()
                ordered_doc_types = []

                for match in matches:
                    doc_type = match.get("doc_type")
                    if doc_type and doc_type not in seen_doc_types:
                        seen_doc_types.add(doc_type)
                        ordered_doc_types.append(doc_type)

                rows.append({
                    "document name": doc_name,
                    "doc_type": ", ".join(ordered_doc_types)
                })
                
                
                
            except Exception as e:
                logging.error(f"[Index {i}] Error processing entry in extract_doc_type_data: {str(e)}")
                traceback.print_exc()

        return pd.DataFrame(rows)

    def resulting_doc_type(self, state: documenttag) -> documenttag:
        
        """
        Determines the document type for each file in the provided state.

        This method processes each document in `state.text_df` by:
          1. Retrieving the document name and content.
          2. Inferring the audience using `get_audience(file_name)`.
          3. Classifying the document type using `get_doc_type(content, audience)`.
          4. Extracting structured output via `extract_doc_type_data()`.
          5. Aggregating the results into a final DataFrame.

        Successfully extracted document types are logged.
        Any errors are logged and collected but do not halt execution.

        Args:
            state (documenttag): A state object containing `text_df` with two columns:
                - Column 0: Document name
                - Column 1: Document content

        Returns:
            documenttag: Updated state object with a new key:
                - `doc_type_df` (DataFrame): Contains the document name and its classified type.
                  The result is also saved to a CSV file named `doc_type_df.csv`.

        Raises:
            Logs exceptions per document without interrupting the iteration.
        """
        
        result = []
        errors = []

        for index, row in state.text_df.iterrows():
            time.sleep(1.5)
            file_name = row[0]  # or row['file'] if you renamed columns
            file_content = row[1]

            try:
                audience = self.get_audience(file_name)
                doc_response = self.get_doc_type(file_content, audience)
                dataframe = self.extract_doc_type_data(doc_response)

                result.append(dataframe)
                
                logging.info(f"[{file_name}] Document type extracted successfully.")
                
            except Exception as e:
                error_message = f"[{file_name}] Error: {str(e)}"
                
                logging.info(error_message)
                
                traceback.print_exc()
                errors.append({"file_name": file_name, "error": str(e)})

        if result:
            doc_type_df = pd.concat(result, ignore_index=True)
        else:
            doc_type_df = pd.DataFrame(columns=["document name", "doc_type"])

        # Optional: save to CSV
        #doc_type_df.to_csv("doc_type_df.csv", index=False)
        
        # Save locally
        try:
            doc_type_df.to_csv("doc_type_df.csv", index=False)
        except Exception as e:
            logging.warning(f"Failed to write doc_type_df.csv: {str(e)}")
        

        # Attach to state and return
        return {"doc_type_df": doc_type_df}

    
    ###################Account Focus#################
    
    def find_account_focus_terms(self, document_name: str, extracted_text: str) -> dict:
        """
        Searches for known account focus terms in the extracted text and returns
        structured output as a Python dictionary (not a JSON string).
        """
        
        """
        Identifies account focus terms within the provided extracted text.

        This method scans the input text for the presence of known account-specific
        terms (e.g., distributor) using case-insensitive exact word matching.
        Matches are collected and returned in a structured dictionary.

        Args:
            document_name (str): The name of the document being analyzed.
            extracted_text (str): The full text content extracted from the document.

        Returns:
            dict: A dictionary with the following structure:
                {
                    "document name": <document_name>,
                    "matches": [<list of matched terms>]
                }
        """
        
        try:
            terms = [
                "Aramark", "Avendra", "Compass", "GFS", "Performance Foods",
                "Premier", "Sodexo", "Sysco", "Unipro", "USF"
            ]

            found_terms = []
            for term in terms:
                pattern = rf'\b{re.escape(term)}\b'
                if re.search(pattern, extracted_text, flags=re.IGNORECASE):
                    found_terms.append(term)

            return {
                "document name": document_name,
                "matches": found_terms
            }
        
        except Exception as e:
            logging.error(f"[{document_name}] Error in find_account_focus_terms: {str(e)}")
            return {
                "document name": document_name if isinstance(document_name, str) else "Unknown Document",
                "matches": []
            }

    def get_account_focus(self, combined_text, audience):
        
        """
        Analyze input content and identify the type of the document using Gemini.

        Args:
            content (str): The input text to identify the type of the document.

        Returns:
            str: The identified account focus, or "No industry segment detected".
        
        Raises:
            Exception: If the Gemini model fails to process the input.
        """
        system_prompt= """
        You are a highly reliable and detail-oriented document analyst with expertise in reviewing business documents such as reports, presentations, contracts, and data exports.
Your primary responsibility is to detect mentions of specific company names or acronyms in unstructured text.
You return results in a structured JSON format, clearly listing any matched terms.
You only include matches where the company name or acronym appears clearly and completely. Do not make assumptions or partial matches.
Always follow the output format strictly as provided by the user.

You are given the name of a document and a block of text extracted from it.
Your task is to check whether any of the following terms are mentioned at least once in the text (case-insensitive, exact match, including acronyms):
1. Aramark  
2. Avendra  
3. Compass  
4. GFS (short for Gordon Foodservice)  
5. Performance Foods  
6. Premier  
7. Sodexo  
8. Sysco  
9. Unipro  
10. USF (short for US Foods)  

### OBJECTIVES:
1. **Account Focus**:
 - Detect and return any mentions of specific company names or acronyms in the given input text. 
 - Do not include duplicates for the same account focus type. Each account focus name or acronym should appear only once in the `matches` array, even if mentioned multiple times in the text.

For each match found, return the term as the "Account Focus".
"""
    
        user_prompt=f"""
        Input text: {combined_text}

        In the above text, look for the occurrences of the specified terms.
        For each match found, return the term as the "Account Focus". Make sure to include each account focus type only once in the json output.

        Output Format (in JSON):
        [
          {{
            "document name": "Name of Document",
            "matches": [
              {{
                "Account Focus": "Name of the Account Focus type"
              }},
              ...
            ]
          }}
        ]

        If no matches are found, return the same structure with an empty 'matches' array:
        [
          {{
            "document name": "Name of Document",
            "matches": []
          }}
        ]

        Only return the JSON. Do not include explanations, summaries, or confirmations.

        Return only a valid JSON output.

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
            
            # print("response: ", response)
            
            logging.info("response: ", response)
            
            account_focus = json.loads(response)
            return account_focus

        except Exception as e:
            raise Exception(f"Error during account focus analysis: {type(e).__name__} - {e}")
    
    
    def extract_acc_type_data(self, json_data):
        """
        Extracts account focus data into a DataFrame.

        Parameters:
            json_data (dict or list): A single dictionary or list of dictionaries,
                                      each with 'document name' and 'matches' (list of strings).

        Returns:
            pd.DataFrame: DataFrame with columns:
                          'document name', 'account focus'
        """
        
        try:
            
            # Normalize input to a list if it's a single dictionary
            if isinstance(json_data, dict):
                json_data = [json_data]

            rows = []

            for entry in json_data:
                
                try:
                    if not isinstance(entry, dict):
                        continue  # Skip invalid entries

                    doc_name = entry.get("document name", "Unknown Document")
                    matches = entry.get("matches", [])

                    # Remove duplicates while preserving order
                    seen = set()
                    ordered_focus_terms = []
                    for term in matches:
                        if term not in seen:
                            seen.add(term)
                            ordered_focus_terms.append(term)

                    rows.append({
                        "document name": doc_name,
                        "account focus": ", ".join(ordered_focus_terms)
                    })
                    
                except Exception as e:
                    logging.error(f"[Entry {idx}] Error processing entry: {str(e)}")
                    continue  # Skip to next entry on failure

            return pd.DataFrame(rows)
        
        except Exception as e:
            logging.critical(f"extract_acc_type_data failed: {str(e)}")
            return pd.DataFrame(columns=["document name", "account focus"])

    
    def resulting_account_focus(self, state: documenttag) -> documenttag:
        
        """
        Processes each document to extract account focus terms and stores the results in a structured DataFrame.

        This method iterates over rows in the state's `text_df`, identifies known account focus terms
        from the document content, logs results or errors, and builds a final DataFrame with matched account terms.

        Args:
            state (documenttag): An object containing a `text_df` DataFrame with document names and extracted text.

        Returns:
            documenttag: A dictionary with the following key:
                - "account_focus_df": A pandas DataFrame with columns:
                    - "document name"
                    - "account focus" (list of matched account terms, if any)
        """
        
        logging.info("Running resulting_account_focus")
        
        result = []
        errors = []

        for index, row in state.text_df.iterrows():
            time.sleep(1.5)
            file_name = row[0]  # or row['file'] if renamed
            
            file_content = row[1]

            try:
                audience = self.get_audience(file_name)
                
                response = self.find_account_focus_terms(file_name, file_content)
                
                logging.info("response resulting_account_focus: ",response)
                logging.info("type resulting_account_focus: ",type(response))
                
                dataframe = self.extract_acc_type_data(response)

                result.append(dataframe)
                
                logging.info(f"[{file_name}] Account focus extracted successfully.")
                
            except Exception as e:
                error_message = f"[{file_name}] Error: {str(e)}"
                
                logging.info(error_message)
                traceback.print_exc()
                errors.append({"file_name": file_name, "error": str(e)})

        try:
            if result:
                account_focus_df = pd.concat(result, ignore_index=True)
            else:
                account_focus_df = pd.DataFrame(columns=["document name", "account focus"])
            
            '''
            # Save to CSV
            account_focus_df.to_csv("account_focus_df.csv", index=False)
            '''
            
        except Exception as final_e:
            logging.critical(f"Failed to compile or write final account focus DataFrame: {str(final_e)}")
            traceback.print_exc()
            account_focus_df = pd.DataFrame(columns=["document name", "account focus"])
        
        # Attach to state and return
        return {"account_focus_df": account_focus_df}


    def update_account_focus(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Updates the 'account focus' column based on a set of business rules:
            1. If 'account focus' is blank, default to "Distributor" or "Operator".
            2. If 'industry segment' is K12 → "Operator"
            3. If 'doc_type' is 'external' and document looks like a selling document → "Distributor"
            4. Otherwise → "Operator"

        Args:
            df (pd.DataFrame): Input DataFrame with 'account focus', 'industry segment', 'doc_type', and 'document name' columns.

        Returns:
            pd.DataFrame: Updated DataFrame with revised 'account focus'.
        """
        
        logging.info("Printing in update_account_focus")

        def classify_account_focus(row):
            
            try:
                logging.info("Printing in classify_account_focus")

                current_focus = str(row.get("account focus", "")).strip()
                industry = str(row.get("industry segment", "")).strip().lower()
                doc_type = str(row.get("doc_type", "")).strip().lower()
                doc_name = str(row.get("document name", "")).strip().lower()

                # Keep non-blank values unchanged
                if current_focus:
                    return current_focus

                # Rule 2: K12 implies Operator
                if industry == "k12":
                    return "Operator"

                # Rule 3: External selling document implies Distributor
                selling_keywords = ["sell", "selling", "promotion", "deal", "offer", "rebate", "discount"]
                if doc_type == "external" and any(keyword in doc_name for keyword in selling_keywords):
                    return "Distributor"

                # Rule 4: Default to Operator
                return "Operator"
            
            except Exception as e:
                logging.error(f"Error classifying row for document '{row.get('document name', 'Unknown')}': {e}")
                return row.get("account focus", "") or "Operator"

        df["account focus"] = df.apply(classify_account_focus, axis=1)
        
        logging.info("End of classify_account_focus")
        
        try:
            df["account focus"] = df.apply(classify_account_focus, axis=1)
            logging.info("Account focus classification completed.")
        except Exception as e:
            logging.critical(f"Failed to apply classification logic: {e}")
            traceback.print_exc()
        
        return df

    def post_processing(self, state: documenttag):
        """
        Merges industry, UPC/BPC, document type, and account focus data on document name.

        Returns:
            documenttag: Updated state with final_df set to the merged result.
        """
        
        logging.info("Printing in post_processing")
        
        try:
            # Define expected DataFrames in state
            sources = {
                "industry_df": state.industry_df,
                "bpc_df": state.bpc_df,
                "doc_type_df": state.doc_type_df,
                "account_focus_df": state.account_focus_df,
            }

            # Log type and preview for each dataframe
            for name, df in sources.items():
                # print(f"\n{name} type:", type(df))

                logging.info(f"\n{name} type:", type(df))

                if isinstance(df, pd.DataFrame):
                    print(f"\n--- {name} HEAD ---")
                    # print(df.head())

                    logging.info(f"\n--- {name} HEAD ---")
                else:
                    raise TypeError(f"Error: {name} is not a DataFrame")

                if "document name" not in df.columns:
                    raise ValueError(f"Error: 'document name' column missing in {name}. Columns found: {df.columns.tolist()}")

            # Merge all DataFrames sequentially on 'document name'
            merged_df = sources["account_focus_df"]
            for name in [ "doc_type_df","bpc_df", "industry_df"]:
                merged_df = pd.merge(merged_df, sources[name], on="document name", how="outer")

            # Fill missing columns
            required_columns = [
                "industry segment",
                "industry sub-segment",
                "base product codes",
                "doc_type",
                "account focus"
            ]
            for col in required_columns:
                if col not in merged_df.columns:
                    merged_df[col] = ""
                else:
                    merged_df[col] = merged_df[col].fillna("")


            logging.info("\n=== post_processing completed successfully ===")
            logging.info("Merged dataframe saved to 'merge_results.csv'")
            logging.info("Final dataframe shape:", merged_df.shape)

            merged_df=self.update_account_focus(merged_df)
            
            #print("merged_df: ", merged_df.head(5))
            
            # Loop over each row and convert to dictionary
            for index, row in df.iterrows():
                
                db_dict = row.to_dict()
                
                print("db_dict: ", db_dict)
                
                insert_initial_row(output_table_name, db_dict)
                
            # Save to CSV
            merged_df.to_csv("merge_results.csv", index=False)
            

            logging.info("End of post_processing")

            return {"final_df": merged_df}
        
        except Exception as e:
            logging.error(f"Exception in post_processing: {e}")
            traceback.print_exc()
            # Return empty DataFrame or partial results as fallback
            return {"final_df": pd.DataFrame()}
    
    