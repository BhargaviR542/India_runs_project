class ProfileNormalizer:
    """Converts multi-structured candidate records and JDs into dense, semantic text blocks.
    
    This process strips formatting and structures text so that embedding models focus on 
    underlying intent and capability rather than keyword density.
    """

    @staticmethod
    def flatten_candidate(candidate: dict) -> str:
        # Extract core background
        skills = ", ".join(candidate.get("skills", []))
        
        # Unpack chronological professional histories
        history_blocks = []
        for exp in candidate.get("career_history", []):
            role_str = f"Role: {exp.get('title', 'Staff')} at {exp.get('company', 'Organization')}. " \
                       f"Scope: {exp.get('description', '')}."
            history_blocks.append(role_str)
        history_combined = " ".join(history_blocks)
        
        # Inject performance metrics and platform signals 
        activity = str(candidate.get("platform_activity", "Standard activity score."))
        
        # Build optimized composite string
        normalized_text = (
            f"Candidate Overview. Core Tech Proficiencies: {skills}. "
            f"Professional Directives & Track Record: {history_combined} "
            f"Behavioral Vectors & Platform Signals: {activity}."
        )
        return normalized_text

    @staticmethod
    def flatten_jd(jd: dict) -> str:
        title = jd.get("title", "Target Role")
        desc = jd.get("description", "")
        req_skills = ", ".join(jd.get("required_skills", []))
        
        return (
            f"Job Mandate Requirements. Target Title: {title}. "
            f"Core Functional Domain Work: {desc}. "
            f"Mandatory Domain Skills Needed: {req_skills}."
        )