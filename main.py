import config
import cohere
import guardrails as gd
from guardrails.validators import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List
from flask import request, jsonify, Flask

# Flask for endpoints
app = Flask(__name__)

# For Pydantic Schema
class Sunscreen(BaseModel):
    name: str = Field(..., description="Name of the sunscreen")
    spf: str = Field(
        ..., 
        description="What is the SPF of the sunscreen?", 
        validators=[ValidChoices(["15", "30", "45", "50"], on_fail="reask")]
    )
    explanations: str = Field(..., description="Why does the sunscreen work well for patient's skin type and location?")

# Flask endpoint to produce a sunscreen recommendation
@app.route('/recommend-sunscreen', methods=['POST'])
def handle_reccomend_sunscreen():
    try:
        skin_type = request.json.get('skin_type')
        complexion = request.json.get('complexion')
        location = request.json.get('location')

        response = recommend_sunscreen(skin_type, complexion, location)

        return jsonify({'recommendation': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# calling the Cohere /chat endpoint to generate a response
def recommend_sunscreen(complexion, skin_type, location):

    # setting up the connection to Cohere API
    co = cohere.Client(config.api_key)

    recommendation = f'''You are a dermatologist. Given the following notes about a user,
        please reccomend a sunscreen that suits their skin and location needs and provide the
        sunscreen name, spf, and an explanation justifying the recommended choice.
        
        Fitzpatrick Skin Type: {complexion}
        Skin Type: {skin_type}
        Location: {location}'''
    
    PROMPT = """Given the following instructions,
    please extract a dictionary that contains the sunscreens's information.

    ${recommendation}
    ${gr.complete_json_suffix_v2}
    """

    # using Guardrails AI for structured output, similar to Pydantic
    guard = gd.Guard.from_pydantic(Sunscreen, prompt=PROMPT)
    print(guard.base_prompt)

    raw_llm_output, validated_ouput = guard(
        co.chat,
        prompt_params={"recommendation": recommendation},
        model='command',
        connectors=[{"id": "web-search"}], # can make custom connectors later
        prompt_truncation="AUTO",
        temperature=0.2 # higher = more random
    )
    print(validated_ouput)

    # to look at the validation process behind the scenes?
    # print(guard.state.most_recent_call.history[0].rich_group)
    # guard.guard_state.most_recent_call.tree


# def calculate_interval(region, spf):
# connect/relate to another API to find the UV index and compare with spf index
# to calculate how often to send a notification through flutter!

### for testing with output structure ###
recommend_sunscreen("Type 1", "dry", "San Francisco")
# print(recommend_sunscreen("Type 3", "oily", "Toronto"))