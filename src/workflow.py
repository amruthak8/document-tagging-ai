import concurrent.futures
from langgraph.graph import StateGraph, START, END

import logging

from src.taging_agent import documenttag, documenttag_segment

def doctagging_workflow(path) -> StateGraph:
    
    print("Printing in doctagging_workflow")
    
    logging.info("Printing in doctagging_workflow")
    
    # Create an instance of Document Tagging
    doctag_inst = documenttag_segment(path)
    
    print("Printing in Langgraph")    
    logging.info("Printing in Langgraph") 
    # Define the LangGraph Workflow
    doctag_workflow = StateGraph(doctag_inst)
    doctag_workflow.add_node("document_reader", doctag_inst.extract_data)
    doctag_workflow.add_node("base_product_code", doctag_inst.base_product_code_tag)
    doctag_workflow.add_node("industry_segment", doctag_inst.industry_segment_tag)
    doctag_workflow.add_node("resulting_doc_type", doctag_inst.resulting_doc_type)
    doctag_workflow.add_node("resulting_account_focus", doctag_inst.resulting_account_focus)
    doctag_workflow.add_node("post_processing", doctag_inst.post_processing)
    
    # Define the flow
    doctag_workflow.add_edge(START, "document_reader")
    doctag_workflow.add_edge("document_reader", "base_product_code")
    doctag_workflow.add_edge("document_reader", "industry_segment")
    doctag_workflow.add_edge("document_reader", "resulting_doc_type")
    doctag_workflow.add_edge("document_reader", "resulting_account_focus")
    doctag_workflow.add_edge("base_product_code", "post_processing")
    doctag_workflow.add_edge("industry_segment", "post_processing")
    doctag_workflow.add_edge("resulting_doc_type", "post_processing")
    doctag_workflow.add_edge("resulting_account_focus", "post_processing")
    doctag_workflow.add_edge("post_processing", END)
    
    logging.info("End of doctagging_workflow")
    return doctag_workflow.compile()


