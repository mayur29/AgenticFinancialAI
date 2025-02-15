import streamlit as st
from pathlib import Path
import tempfile
import time
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo


# Data Structures
@dataclass
class VideoAnalysisConfig:
    """Configuration settings for video analysis"""

    supported_formats: tuple = ("mp4", "mov", "avi")
    max_retries: int = 3
    retry_delay: int = 2
    timeout: int = 300  # 5 minutes


class VideoAnalyzer:
    """Main class for handling video analysis operations"""

    def __init__(self):
        self.config = VideoAnalysisConfig()
        self.setup_environment()
        self.agent = self.initialize_agent()

    def setup_environment(self) -> None:
        """Initialize environment and API configuration"""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        api_key = "AIzaSyCavmIbjwEAakTZkCOgfT3WXzAEV-8M4Ps"
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)

    @staticmethod
    @st.cache_resource
    def initialize_agent() -> Agent:
        """Initialize and cache the AI agent"""
        return Agent(
            name="Video AI Summarizer",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def generate_analysis_prompt(
        self, user_query: str, video_type: str = "general"
    ) -> str:
        """Generate appropriate analysis prompt based on video type"""
        base_prompt = f"""
        As an expert video analyst, I'll examine this video comprehensively and provide detailed insights.
        
        User's Question: "{user_query}"
        
        Analysis Framework:
        1. Content Overview
        - Visual elements and composition
        - Audio components (speech, music, effects)
        - Overall narrative and flow
        - Production quality and style
        
        2. Detailed Analysis
        - Key moments and highlights (with timestamps)
        - Important patterns or themes
        - Technical elements worth noting
        - Contextual significance
        
        3. Specific Response to User Query
        [Provide detailed, evidence-based response to the user's question]
        
        4. Additional Insights
        - Background context and relevance
        - Notable observations
        - Potential applications or implications
        - Related information from web research
        
        Please structure the response as follows:
        
        📽️ VIDEO SUMMARY
        [Brief but comprehensive overview]
        
        🎯 ADDRESSING YOUR QUESTION
        [Direct response with specific examples]
        
        💡 KEY INSIGHTS
        [Notable findings and patterns]
        
        🔍 ADDITIONAL CONTEXT
        [Relevant background and web research]
        
        Keep the analysis:
        - Clear and conversational
        - Well-supported with specific examples
        - Balanced between technical detail and accessibility
        - Focused on practical insights
        """

        # Add video type-specific guidance
        type_specific_prompts = {
            "tutorial": """
            Additional Tutorial-Specific Analysis:
            - Learning objectives and their clarity
            - Step-by-step instruction quality
            - Practical demonstration effectiveness
            - Prerequisites and target audience
            """,
            "presentation": """
            Additional Presentation-Specific Analysis:
            - Slide design and visual aids
            - Speaker delivery and engagement
            - Key message clarity
            - Supporting evidence quality
            """,
            "product": """
            Additional Product-Specific Analysis:
            - Feature demonstration clarity
            - Value proposition communication
            - Technical specifications coverage
            - Comparison with similar products
            """,
        }

        if video_type in type_specific_prompts:
            base_prompt += type_specific_prompts[video_type]

        return base_prompt

    def process_video(self, video_file) -> str:
        """Process uploaded video file and return temporary file path"""
        if not video_file:
            raise ValueError("No video file provided")

        if not video_file.name.lower().endswith(self.config.supported_formats):
            raise ValueError(
                f"Unsupported file format. Please upload: {', '.join(self.config.supported_formats)}"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            return temp_video.name

    def analyze_video(
        self, video_path: str, user_query: str, video_type: str = "general"
    ) -> Dict[str, Any]:
        """Perform video analysis with retry logic and error handling"""
        if not os.path.exists(video_path):
            raise FileNotFoundError("Video file not found")

        attempts = 0
        last_error = None

        while attempts < self.config.max_retries:
            try:
                processed_video = genai.upload_file(video_path)

                # Wait for processing with timeout
                start_time = time.time()
                while processed_video.state.name == "PROCESSING":
                    if time.time() - start_time > self.config.timeout:
                        raise TimeoutError("Video processing timeout")
                    time.sleep(self.config.retry_delay)
                    processed_video = genai.get_file(processed_video.name)

                analysis_prompt = self.generate_analysis_prompt(user_query, video_type)
                response = self.agent.run(analysis_prompt, videos=[processed_video])

                return {
                    "success": True,
                    "content": response.content,
                    "processing_time": time.time() - start_time,
                }

            except Exception as e:
                attempts += 1
                last_error = str(e)
                if attempts < self.config.max_retries:
                    time.sleep(self.config.retry_delay)

        return {
            "success": False,
            "error": f"Analysis failed after {attempts} attempts. Last error: {last_error}",
        }


def setup_streamlit_page():
    """Configure Streamlit page layout and styling"""
    st.set_page_config(page_title="AI Video Analyzer", page_icon="🎥", layout="wide")

    st.title("AI Video Analysis Suite 🎥✨")
    st.header("Powered by Gemini AI")

    # Custom styling
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            color: #721c24;
            margin: 1rem 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point"""
    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer()

        # Setup page
        setup_streamlit_page()

        # Sidebar for settings
        with st.sidebar:
            st.subheader("Analysis Settings")
            video_type = st.selectbox(
                "Video Type",
                ["general", "tutorial", "presentation", "product"],
                help="Select the type of video for more targeted analysis",
            )

            st.markdown("---")
            st.markdown(
                """
                ### Tips for Best Results
                - Ensure good video quality
                - Keep videos under 10 minutes
                - Provide specific questions
                - Select appropriate video type
            """
            )

        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            video_file = st.file_uploader(
                "Upload Your Video",
                type=list(analyzer.config.supported_formats),
                help="Select a video file to analyze",
            )

            if video_file:
                video_path = analyzer.process_video(video_file)
                st.video(video_path)

                user_query = st.text_area(
                    "What would you like to know about this video?",
                    placeholder="Example: What are the main topics covered? How effective is the presentation?",
                    help="Be specific in your question for better results",
                )

                if st.button("🔍 Analyze Video", use_container_width=True):
                    if not user_query:
                        st.warning("Please enter a question about the video.")
                    else:
                        try:
                            with st.spinner(
                                "🎬 Analyzing your video... This may take a few minutes."
                            ):
                                result = analyzer.analyze_video(
                                    video_path, user_query, video_type
                                )

                            if result["success"]:
                                st.success(
                                    f"Analysis completed in {result['processing_time']:.1f} seconds!"
                                )
                                st.markdown("### Analysis Results")
                                st.markdown(result["content"])
                            else:
                                st.error(result["error"])

                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

                        finally:
                            # Cleanup
                            Path(video_path).unlink(missing_ok=True)

        with col2:
            st.markdown(
                """
                ### How It Works
                1. Upload your video
                2. Select video type
                3. Ask your question
                4. Get AI-powered insights
                
                ### Features
                - Multi-format support
                - Detailed analysis
                - Context-aware responses
                - Web research integration
            """
            )

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info(
            "Please refresh the page and try again. If the error persists, check your API key configuration."
        )


if __name__ == "__main__":
    main()
