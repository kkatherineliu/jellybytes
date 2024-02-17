import config
import cohere
# import json
import guardrails as gd
from guardrails.validators import ValidChoices
from pydantic import BaseModel, Field
# from rich import print
from flask import request, jsonify, Flask
import requests

#########################################################################
############################ Flask endpoints ############################
#########################################################################

app = Flask(__name__)

# Find UV Index based on location
@app.route('/uv-index', methods=['GET'])
def get_uv_index():
    try:
        location = request.args.get('location')
        response = find_uv_index(location)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Produce sunscreen recommendation
@app.route('/sunscreen', methods=['POST'])
def create_sunscreen():
    try:
        skin_type = request.json.get('skin_type')
        complexion = request.json.get('complexion')
        location = request.json.get('location')

        response = recommend_sunscreen(skin_type, complexion, location)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Find UV index of their location

########################################################################
################ Finding the UV Index of their location ################
########################################################################
def find_uv_index(location):
    # Define your headers
    headers = {
        "x-access-token": config.api_key_uv,
        "Content-Type": "application/json"
    }
    params = {}

    if (location == "Toronto"):
        params = {
        "lat": 51.5, 
        "lng": -0.11, 
        "alt": 100,
        "dt": "" 
        }
    elif (location == "Vancouver"):
        params = {
        "lat": 51.5, 
        "lng": -0.11,
        "alt": 100,
        "dt": ""
        }
    elif (location == "Montreal"):
        params = {
        "lat": 51.5,
        "lng": -0.11,
        "alt": 100,
        "dt": ""
        }
    else:
        pass # how to return an error so that the Flask thing goes to 500
    
    response = requests.get("https://api.openuv.io/api/v1/uv", headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json() # should i reduce this to only return the UV so its easier for front end?
    else:
        print(f'Error: {response.status_code}', response.text) # not sure if this is the right way to return

# returns the time in minutes before reapplying
def reapply_interval(uv):
    if (uv <= 5):
        return 120
    elif (uv <= 8):
        return 90
    else:
        return 75

# Calculate time until burning based on the SPF they have?
# ^^ might be better to have it in the Dart portion instead of latency with the request since it's a simple calculation anyways
# to calculate how often to send a notification through flutter!
    
########################################################################
############### Generating a recommendation for the user ###############
########################################################################

# Interacting with Cohere /chat endpoint to generate a response
def recommend_sunscreen(complexion, skin_type, location):
    
    # setting up the connection to Cohere API
    co = cohere.Client(config.api_key_cohere)

    recommendation = co.chat(
        message=f'''You are a dermatologist. Given the following notes about a user,
        please recommend ONE non-greasy sunscreen. Indicate the name, SPF,
        and ONE short sentence to explain for why it is suitable their skin and location needs.
        
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
        max_tokens=256,
        temperature=0.2
    )
    # print(validated_response)
    return validated_response # dictionary data type

# JSON Schema following Pydantic form
class Sunscreen(BaseModel):
    name: str = Field(description="Name of the sunscreen")
    spf: str = Field(
        description="What is the SPF of the sunscreen?", 
        validators=[ValidChoices(choices=["15", "30", "45", "50"], on_fail="reask")]
    )
    explanation: str = Field(description="Why does the sunscreen work well for patient's skin type and location?")

### for testing with output structure ###
# print(recommend_sunscreen("Type 1", "dry", "San Francisco"))
print(recommend_sunscreen("Type 3", "dry", "Toronto"))
# print(recommend_sunscreen("Type 2", "normal", "Toronto"))
# recommend_sunscreen("Type 4", "acne-prone", "Australia")