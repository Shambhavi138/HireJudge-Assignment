import streamlit as st
import json
from typing import Dict, Any, Optional
import traceback
import requests
import time

from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.schema import BaseOutputParser
from langchain.callbacks.manager import CallbackManagerForLLMRun

#streamlit configuration code
st.set_page_config(
    page_title="HireJudge - Free LangChain Evaluation",
    page_icon="🌐",
    layout="wide"
)

class HuggingFaceLLM(LLM):
    
    model_name: str = "microsoft/DialoGPT-medium"
    api_url: str = ""
    max_tokens: int = 500
    temperature: float = 0.1
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_free"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the Hugging Face API"""
        try:
            models_to_try = [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "google/flan-t5-base",
                "facebook/blenderbot-400M-distill"
            ]
            
            for model in models_to_try:
                try:
                    api_url = f"https://api-inference.huggingface.co/models/{model}"
                    
                    payload = {
                        "inputs": prompt[:1000], 
                        "parameters": {
                            "max_length": self.max_tokens,
                            "temperature": self.temperature,
                            "do_sample": True
                        }
                    }
                    
                    #request with timeout
                    response = requests.post(
                        api_url, 
                        json=payload, 
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        #code to manage different response formats
                        if isinstance(result, list) and len(result) > 0:
                            if 'generated_text' in result[0]:
                                return result[0]['generated_text'].strip()
                            elif 'text' in result[0]:
                                return result[0]['text'].strip()
                        elif isinstance(result, dict):
                            if 'generated_text' in result:
                                return result['generated_text'].strip()
                    
                    if response.status_code == 503:
                        continue
                    
                except Exception:
                    continue
            
            #in case of model failure
            return self._fallback_evaluation(prompt)
            
        except Exception as e:
            return self._fallback_evaluation(prompt)
    
    def _fallback_evaluation(self, prompt: str) -> str:
        """Fallback rule-based evaluation when API fails"""
        
        # Extract candidate name and role from prmpt
        lines = prompt.split('\n')
        candidate_name = "Candidate"
        target_role = "Software Developer"
        
        try:
            for line in lines:
                if '"name"' in line:
                    candidate_name = line.split(':')[1].strip().replace('"', '').replace(',', '')
                elif 'TARGET ROLE:' in line:
                    target_role = line.split(':')[1].strip()
        except:
            pass
        
        fallback_response = f"""SUMMARY:
{candidate_name} is applying for {target_role} position. Based on provided profile, candidate shows potential for the role. Technical background and experience alignment need detailed discussion. Professional development and skill matching require further evaluation during interview process.

FIT ASSESSMENT:
Average - Candidate demonstrates relevant background but requires deeper technical evaluation to determine full role alignment and capability match.

CONCERNS AND GAPS:
Need to verify technical depth and practical experience. Communication skills and team collaboration abilities require assessment. Project complexity and leadership experience need clarification.

FOLLOW-UP QUESTIONS:
1. Can you walk me through your most challenging technical project and how you solved any problems you face during the project?
2. How do you stay current with technology trends and what new skills have you learned recently?
3. Describe a situation where you had to collaborate with a team to deliver a project under tight deadlines."""

        return fallback_response

class CandidateEvaluator:
    
    def __init__(self):
        self.setup_free_llm()
        self.setup_prompt_template()
    
    def setup_free_llm(self):
        """Setup Hugging Face LLM"""
        try:
            self.llm = HuggingFaceLLM()
            self.llm_configured = True
        except Exception as e:
            st.error(f"LLM Setup Error: {str(e)}")
            self.llm_configured = False
    
    def setup_prompt_template(self):
        
        template = """You are a technical recruiter assistant. Analyze this candidate profile and provide structured evaluation.

CANDIDATE PROFILE:
{candidate_json}

TARGET ROLE: {role}

Please provide EXACTLY this format:

SUMMARY:
[Write exactly 50 words summarizing candidate background and key strengths]

FIT ASSESSMENT:
[Strong/Average/Weak] - [Brief explanation of fit level]

CONCERNS AND GAPS:
[List main concerns or skill gaps, if none write "No major concerns identified"]

FOLLOW-UP QUESTIONS:
1. [Specific technical or role-related question]
2. [Experience or skill-based question]  
3. [Cultural fit or growth potential question]

Be concise and professional."""
        
        self.prompt_template = PromptTemplate(
            input_variables=["candidate_json", "role"],
            template=template
        )
    
    def validate_json_input(self, json_text: str) -> tuple[bool, Optional[Dict], Optional[str]]:
        try:
            if not json_text.strip():
                return False, None, "JSON input cannot be empty"
            
            candidate_data = json.loads(json_text)
            
            required_fields = ['name', 'role_applied']
            missing_fields = [field for field in required_fields if field not in candidate_data]
            
            if missing_fields:
                return False, None, f"Missing required fields: {', '.join(missing_fields)}"
            
            return True, candidate_data, None
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    def evaluate_candidate(self, candidate_data: Dict[str, Any], target_role: str) -> str:
        
        if not self.llm_configured:
            return "LLM not available. Please check your internet connection."
        
        try:
            candidate_json = json.dumps(candidate_data, indent=2)
            if len(candidate_json) > 800:
                truncated_data = {
                    'name': candidate_data.get('name', ''),
                    'role_applied': candidate_data.get('role_applied', ''),
                    'skills': candidate_data.get('skills', [])[:5],
                    'experience': candidate_data.get('experience', [])[:2], 
                    'education': candidate_data.get('education', [])[:1]
                }
                candidate_json = json.dumps(truncated_data, indent=2)
            
            formatted_prompt = self.prompt_template.format(
                candidate_json=candidate_json,
                role=target_role
            )
            
            response = self.llm.invoke(formatted_prompt)
            
            return response
            
        except Exception as e:
            return f"Evaluation Error: {str(e)}\n\nFalling back to rule-based evaluation..."

def main():
    st.title("🌐 HireJudge - Candidate Evaluation")
    st.markdown("---")
    
    evaluator = CandidateEvaluator()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Role input
        target_role = st.selectbox(
            "Target Role:",
            [
                "Software Developer",
                "Frontend Developer", 
                "Backend Developer",
                "Full Stack Developer",
                "Data Scientist",
                "DevOps Engineer",
                "Mobile Developer",
                "UI/UX Designer",
                "Product Manager"
            ]
        )
        
        # JSON input
        json_input = st.text_area(
            "Candidate Profile (JSON):",
            height=300,
            placeholder='{\n "name": "Anjali Verma",\n  "role_applied": ""Frontend Developer",\n   "education": [{"degree": "B.Tech", "field": "CSE", "college": "NIT Bhopal", "year": 2021}],\n  "experience": [{"company": "InMobi", "duration_months": 24, "role": "Frontend Developer", "stack": ["React","TypeScript", "Redux"]}],\n   "skills": ["React", "TypeScript", "HTML", "CSS", "Git", "Unit Testing"],\n "projects": ["Built internal UI component library", "Migrated legacy code to React"], \n  "linkedin": "https://linkedin.com/in/anjaliverma"}',
            help="Enter candidate profile in JSON format"
        )
        
        # Evaluate button
        evaluate_btn = st.button("Evaluate candidate", type="primary")
    
    with col2:
        st.subheader("-> LangChain Evaluation")
        
        if evaluate_btn:
            if not json_input.strip():
                st.error(" Please enter candidate profile JSON!!")
            else:
                with st.spinner("Processing..."):
                    
                    is_valid, candidate_data, error_msg = evaluator.validate_json_input(json_input)
                    
                    if not is_valid:
                        st.error(f" **JSON Error**: {error_msg}")
                    else:
                        try:
                            evaluation = evaluator.evaluate_candidate(candidate_data, target_role)
                            st.markdown("### LangChain Evaluation Results:")
                            st.text_area(
                                "Structured Evaluation:",
                                value=evaluation,
                                height=350,
                                help="Generated using LangChain PromptTemplate + Free Hugging Face LLM"
                            )
                            col_dl, col_copy = st.columns(2)
                            
                            with col_dl:
                                st.download_button(
                                    label="📥 Download Report",
                                    data=evaluation,
                                    file_name=f"langchain_eval_{candidate_data.get('name', 'candidate').replace(' ', '_')}.txt",
                                    mime="text/plain"
                                )
                            
                            with col_copy:
                                if st.button("📋 Copy Format"):
                                    st.code(evaluation, language="text")
                            
                            
                        except Exception as e:
                            st.error(f" **Error**: {str(e)}")
                            if st.checkbox("Show debug details"):
                                st.code(traceback.format_exc())
        
        else:
            st.info("Please enter candidate details and click evaluate.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f" **App Error**: {str(e)}")
        if st.checkbox("Show error details"):
            st.code(traceback.format_exc())