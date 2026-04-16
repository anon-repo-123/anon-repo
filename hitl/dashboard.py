"""
Human-in-the-Loop Dashboard for reviewing uncertain responses.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import PipelineOrchestrator


# Page configuration
st.set_page_config(
    page_title="HITL Review Dashboard",
    page_icon="👁️",
    layout="wide"
)

st.title("Human-in-the-Loop Review Dashboard")
st.markdown("Review and adjudicate uncertain responses from the multi-agent system.")


def load_pipeline():
    """Load the pipeline orchestrator."""
    pipeline = PipelineOrchestrator()
    pipeline.load_hitl_queue()
    return pipeline


def save_review(pipeline, review_data):
    """Save a review and update the queue."""
    hitl_queue = pipeline.hitl_queue
    
    for i, item in enumerate(hitl_queue):
        if item["query_id"] == review_data["query_id"]:
            hitl_queue[i]["status"] = "reviewed"
            hitl_queue[i]["reviewed_at"] = datetime.now().isoformat()
            hitl_queue[i]["human_decision"] = review_data["decision"]
            hitl_queue[i]["human_response"] = review_data.get("response", "")
            hitl_queue[i]["reviewer_notes"] = review_data.get("notes", "")
            break
    
    # Save updated queue
    with open("results/hitl_queue.json", "w") as f:
        json.dump(hitl_queue, f, indent=2)
    
    # Also save to reviewed file
    reviewed_file = "results/hitl_reviewed.json"
    if os.path.exists(reviewed_file):
        with open(reviewed_file, 'r') as f:
            reviewed = json.load(f)
    else:
        reviewed = []
    
    reviewed.append({
        "query_id": review_data["query_id"],
        "reviewed_at": datetime.now().isoformat(),
        "decision": review_data["decision"],
        "human_response": review_data.get("response", ""),
        "notes": review_data.get("notes", "")
    })
    
    with open(reviewed_file, 'w') as f:
        json.dump(reviewed, f, indent=2)
    
    st.success(f"Review saved for query {review_data['query_id']}")


def main():
    pipeline = load_pipeline()
    pending = pipeline.get_pending_reviews()
    
    st.sidebar.header("Dashboard Stats")
    st.sidebar.metric("Pending Reviews", len(pending))
    
    reviewed_count = 0
    reviewed_file = "results/hitl_reviewed.json"
    if os.path.exists(reviewed_file):
        with open(reviewed_file, 'r') as f:
            reviewed_count = len(json.load(f))
    st.sidebar.metric("Completed Reviews", reviewed_count)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Review Triggers")
    st.sidebar.markdown("""
    - **Confidence < 0.5**: Human provides answer directly
    - **Confidence 0.5-0.75**: Human reviews gatekeeper decision
    - **Faithfulness < 0.70**: Human adjudicates correctness
    - **Editor removal > 50%**: Human checks for information loss
    """)
    
    if not pending:
        st.info("No pending reviews. All responses have been processed automatically.")
        return
    
    # Display pending reviews
    for idx, item in enumerate(pending):
        with st.expander(f"Query {idx + 1}: {item['query'][:100]}...", expanded=idx == 0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Query")
                st.write(item['query'])
                
                st.markdown("#### Candidate Answer")
                st.text_area("Generated answer", item['candidate_answer'], height=150, key=f"candidate_{idx}")
                
                st.markdown("#### Edited Answer")
                st.text_area("After editor", item['edited_answer'], height=150, key=f"edited_{idx}")
            
            with col2:
                st.markdown("#### Review Context")
                
                # Show trigger reason
                trigger_reason = item.get('review_reason', 'unknown')
                trigger_colors = {
                    'low_confidence_below_0.5': '🔴',
                    'medium_confidence_requires_review': '🟡',
                    'low_faithfulness': '🟠',
                    'excessive_removal': '🔵'
                }
                st.markdown(f"**Trigger:** {trigger_colors.get(trigger_reason, '⚪')} {trigger_reason.replace('_', ' ').title()}")
                
                st.metric("Gatekeeper Confidence", f"{item.get('gatekeeper_confidence', 0):.2f}")
                st.metric("Verifier Faithfulness", f"{item.get('verifier_faithfulness', 0):.2f}")
                st.metric("Editor Removal %", f"{item.get('removal_percentage', 0)*100:.1f}%")
            
            st.markdown("---")
            st.markdown("#### Human Adjudication")
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                decision = st.radio(
                    "Decision",
                    ["Accept", "Reject", "Modify"],
                    key=f"decision_{idx}"
                )
            
            with col_b:
                if decision == "Modify":
                    human_response = st.text_area(
                        "Provide corrected response:",
                        value=item['edited_answer'],
                        height=100,
                        key=f"response_{idx}"
                    )
                else:
                    human_response = item['edited_answer'] if decision == "Accept" else ""
            
            notes = st.text_input("Reviewer notes (optional)", key=f"notes_{idx}")
            
            if st.button(f"Submit Review for Query {idx + 1}", key=f"submit_{idx}"):
                review_data = {
                    "query_id": item["query_id"],
                    "decision": decision.lower(),
                    "response": human_response if decision == "Modify" else (item['edited_answer'] if decision == "Accept" else ""),
                    "notes": notes
                }
                save_review(pipeline, review_data)
                st.rerun()


if __name__ == "__main__":
    main()
