from dotenv import load_dotenv
import os
import json

#better than llm -> grow
#agnent orch -> crewai  best -> langgraph - also agnet in langraph 
from crewai import Agent, Task, Crew, Process
from datetime import datetime
from pathlib import Path
from utils import (
    load_memory,
    save_memory,
    format_memory_for_context,
    extract_text_from_pdf
)

load_dotenv()

class DocumentProcessor:
    ALLOWED_DIR = Path(".").resolve()
    
    def __init__(self):
        self.memory_file = "memory.json"
        self.memory = load_memory(self.memory_file)
        
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
        self.llm = "gemini/gemini-2.5-flash"
        
        self.extraction_agent = self.create_extraction_agent()
        self.validation_agent = self.create_validation_agent()
        self.reflection_agent = self.create_reflection_agent()
    
    def create_extraction_agent(self):
        return Agent(
            role= "Paytstub PDF Data Extractor",
            goal="Exatract structured data from paystub PDF docuemnt while maintaining accuracy and consistency",
            backstory="Expert in document analysis with attention to detail and pattern recognistion and leraning from mistakes",
            llm = self.llm,
            verbose = True
        )
        
    def create_extraction_task(self, document_text):
        memory_context = format_memory_for_context(self.memory)
        return Task(
            description=f"""Extract a paystub pdf file for its data from this document

{memory_context}

Document:{document_text}

Return only Valid JSON:
{{
    "employee_name":"string",
    "pay_period_start":"YYYY-MM-DD",
    "pay_period_end":"YYYY-MM-DD",
    "pay_date":"YYYY-MM-DD",
    "gross_pay":"number",
    "net_pay":"number",
    "total_deductions":"number",
    "ytd_gross":"number",
    "confidence":"0.0-1.0"
}}
Look for explicit lables like "Pay Date",
"Employee Name", "Period" etc.
If you cannot find a value, use null.
Do not Hallucinate or make up values.
Only valid JSON output""",
            agent = self.extraction_agent,
            expected_output="Structured JSON data containing all key paystub information with validation flags"
        )
        
    def create_validation_agent(self):
        return Agent(
            role = "Data Validator",
            goal = """Validate ectracted data for accuracy and completeness against 
            expected formats""",
            backstory = """Detail-oriented expert who cross-check information and 
            flags inconsistencies or missing data points that may identifies""",
            llm = self.llm,
            verbose = True #logging -> crewai logging
        )
    
    def create_validation_task(self,extraction_result):
        return Task(
            description = f"""Validate this payustub.
{extraction_result}

CHECK:
1. pay_date >= pay_period_start
2. pay_date >= pay_period_end
3. gross_pay >= net_pay 
4. gross_pay = net_pay + total_deductions
5. Confidence >=0.70 

Return JSON :
{{
    "is_valid":true/false,
    "error":[{{"field":"name", "message":""error}}]
}}
""",
            agent = self.validation_agent,
            expected_output = "Validation reports with accuracy scores and correction recommendations"
        )
    
    def create_reflection_agent(self):
        return Agent(
            role = "Learning Reflector or Learning Analyst",
            goal = "Analyze validation feedback and update knowledge base with correction",
            backstory = "Agent that learns from errors and improves extarction patterns over time",
            llm = self.llm,
            verbose = True
        )
    
    #llm -> rules 
    def create_reflection_task(self,document_text, extraction, errors):
        return Task(
            description = f"""Analyze paystub extraction mistakes
DOCUMENT:{document_text}
EXTRACTION RESULT :{extraction}
VALIDATION ERRORS : {errors}

RETURN JSON:

{{
    "mistake_description":"string",
    "correction_rule":"rule to prevent this",
    "confidence":"0.0-1.0"
}}
""",
            agent = self.reflection_agent,
            expected_output = "JSON with mistake_description, correction_rule and confidence"
        )
    #clean your ```
    def _parse_json_from_result(self,result_str):
        result_str = str(result_str)
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()
        elif '```' in result_str:
            result_str = result_str.split('```')[1].split('```')[0].strip()
        return json.loads(result_str)
    
    
    def process_document(self, pdf_path):
        print(f"Processing document: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return None
        
        document_text = extract_text_from_pdf(pdf_path, self.ALLOWED_DIR)
        print(f"Extracted {len(document_text)} characters from PDF")
        
        #Step:1  Extraction agent task
        extraction_task = self.create_extraction_task(document_text)
        extraction_crew = Crew(
            agents=[self.extraction_agent],
            tasks=[extraction_task],
            process=Process.sequential
        )
        extraction_result = extraction_crew.kickoff() #main runs the agent
        print(f"Extraction complete!!")
        
        try:
            extraction_data = self._parse_json_from_result(extraction_result)
        except Exception as e:
            print(f"Error parsing extraction result: {e}")
            return None
        
        #Step:2 Validation agent Task
        validation_task = self.create_validation_task(extraction_data)
        validation_crew = Crew(
            agents=[self.validation_agent],
            tasks=[validation_task],
            process=Process.sequential
        )
        validation_result = validation_crew.kickoff()
        print("Validation Complete!!!")
        
        try:
            validation_data = self._parse_json_from_result(validation_result)
        except Exception as e:
            print(f"Error parsing extraction result: {e}")
            return None
        
        #Step:3 Reflection Agent Task
        reflection_data = None 
        
        if not validation_data.get("is_valid"):
            print(f"Errors found - running self reflection!")
            
            
            reflection_task=self.create_reflection_task(document_text,
                                                        extraction_data,
                                                        validation_data.get("error",[]))
            reflection_crew = Crew(
                agents=[self.reflection_agent],
                tasks=[reflection_task],
                process=Process.sequential
            )
            reflection_result = reflection_crew.kickoff()
            
            try: 
                reflection_data = self._parse_json_from_result(reflection_result)
                
                #memory 
                learning = {
                    "id":f"{len(self.memory)+1:03d}",
                    "timestamp":datetime.now().isoformat(),
                    "mistake":reflection_data.get("mistake_description"),
                    "correction":reflection_data.get("correction_rule"),
                    "confidence":reflection_data.get("confidence")
                }
                self.memory.append(learning)
                save_memory(self.memory, self.memory_file)
                print(f"Learning saved {learning['correction']}")
            except Exception as e:
                print(f"Error parsing reflection result: {e}") 
        else:
            print("Extraction and Validation Successfull!")
            
        return {
            "extraction":extraction_data,
            "validation":validation_data,
            "reflection":reflection_data,
            "memory_count":len(self.memory)
        }
        
def main():
    processor = DocumentProcessor()
    result = processor.process_document("file.pdf")
    
    if result:
        print("Final Result")
        print(f"Confidence : {result['extraction'].get('confidence', 'N/A')}")
        print(f"Valid : {result['validation'].get('is_valid', 'N/A')}")
        print(f"Memory Count : {result['memory_count']}")
        
        errors = result['validation'].get("error",[])
        if errors:
            print("Errors:")
            for error in errors:
                print(f"- {error.get('field')}: {error.get('message')}")
        if result.get('reflection'):
            print(f"Reflection : {result['reflection'].get('correction_rule','N/A')}")
            
        if processor.memory:
            print("\nRecent learnings")
            for learning in processor.memory[-3:]:
                print(f"- {learning['correction']} (Confidence: {learning['confidence']})")
                
                
if __name__ == "__main__":
    main()
        
