import config
import cohere
# import json
import guardrails as gd
from guardrails.validators import ValidChoices
from pydantic import BaseModel, Field
# from rich import print
from flask import request, jsonify, Flask

# setting up the connection to Cohere API
co = cohere.Client(config.api_key)

#########################################################################
############################ Flask endpoints ############################
#########################################################################

app = Flask(__name__)

# Produce sunscreen recommendation
@app.route('/recommend-sunscreen', methods=['POST'])
def handle_reccomend_sunscreen():
    try:
        skin_type = request.json.get('skin_type')
        complexion = request.json.get('complexion')
        region = request.json.get('region')
        sunlight_exposure = request.json.get('sunlight_exposure')

        response = recommend_sunscreen(skin_type, complexion, region, sunlight_exposure)

        return jsonify({'recommendation': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Find UV index of the area

########################################################################
############### Generating a recommendation for the user ###############
########################################################################

# Interacting with Cohere /chat endpoint to generate a response
def recommend_sunscreen(complexion, skin_type, location):
    recommendation = co.chat(
        message=f'''You are a dermatologist. Given the following notes about a user,
        please recommend ONE sunscreen. Indicate the name, SPF,
        and ONE sentence to explain for why it is suitable their skin and location needs.
        Do not include ant other information or repeat the SPF as part of the description. 
        
        Fitzpatrick Skin Type: {complexion}
        Skin Type: {skin_type}
        Location: {location}''',
        model="command",
        connectors=[{"id": "web-search"}], # for grounded results/reduce hallucinations
        # can make custom connectors later
        prompt_truncation="AUTO",
        temperature=0.2 # higher = more random
    )
    return structure_output(recommendation.text)

# Structuring output as a JSON for easy use with front end
def structure_output(recommendation):
    PROMPT = """Please extract a dictionary that contains the sunscreens's information. 
    ${recommendation}
    ${gr.complete_json_suffix_v2}"""

    # using Guardrails AI for structured output
    guard = gd.Guard.from_pydantic(output_class=Sunscreen, prompt=PROMPT)
    # print(guard.base_prompt)

    raw_llm_response, validated_response, *rest = guard(
        co.generate,
        model="command",
        prompt_params={"recommendation": recommendation},
        max_tokens=50,
        temperature=0.2
    )
    return validated_response # dictionary data type

# JSON Schema following Pydantic form
class Sunscreen(BaseModel):
    name: str = Field(description="Name of the sunscreen")
    spf: str = Field(
        description="What is the SPF of the sunscreen?", 
        validators=[ValidChoices(choices=["15", "30", "45", "50"], on_fail="reask")]
    )
    explanation: str = Field(description="Why does the sunscreen work well for patient's skin type and location?")

# def calculate_interval(region, spf):
# connect/relate to another API to find the UV index and compare with spf index
# to calculate how often to send a notification through flutter!

### for testing with output structure ###
# print(recommend_sunscreen("Type 1", "dry", "San Francisco"))
# print(recommend_sunscreen("Type 3", "oily", "Toronto"))
# print(recommend_sunscreen("Type 2", "normal", "Toronto"))
print(recommend_sunscreen("Type 4", "acne-prone", "Australia"))