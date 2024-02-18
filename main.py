#### imports ####
import config

# LLM sunscreen recommendations
import cohere
import guardrails as gd
from guardrails.validators import ValidChoices
from pydantic import BaseModel, Field

# endpoints and API interaction
from flask import request, jsonify, Flask
from pyowm.owm import OWM

#### constants ####
TEMPERATURE = 0
UV_INDEX = 1

#########################################################################
############################ Flask endpoints ############################
#########################################################################

app = Flask(__name__)

# Find temperature of the location
@app.route('/temperature', methods=['GET']) # not sure how to send arguments over yet
def get_temperature():
    try:
        location = request.args.get('location')
        response = find_weather(location, TEMPERATURE)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Find UV Index based on location
@app.route('/uv-index', methods=['GET'])
def get_uv_index():
    try:
        location = request.args.get('location')
        response = find_weather(location, UV_INDEX)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Determine how often to reapply
@app.route('/reapply-frequency', methods=['GET']) # not sure if i should use get or post
def get_reapply_interval():
    try:
        uv = request.args.get('uv')
        complexion = request.args.get('complexion')
        response = reapply_interval(uv, complexion)
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

########################################################################
################### Finding the weather information ####################
########################################################################

# location is a string of the city in Canada
# type is either TEMPERATURE or UV_INDEX
def find_weather(location, type): ## btw the API doesn't work
    owm = OWM(config.api_key_owm) # still says invalid api key provided
    geo_mgr = owm.geocoding_manager()
    weather_mgr = owm.weather_manager()

    list_of_locations = geo_mgr.geocode(location, country='CA') # just cities in Canada for now
    location_coors = list_of_locations[0]
    one_call = weather_mgr.one_call(lat=location_coors.lat, lon=location_coors.lon, exclude='minutely,hourly,daily', units='metric')
    
    if (type == TEMPERATURE):
        return one_call.current.temperature()
    elif (type == UV_INDEX):
        return one_call.current.uvi()
    
# print(find_weather('Toronto', TEMPERATURE))
# print(find_weather('Toronto', UV_INDEX))

# returns the time in minutes before reapplying
# complexion is one of the 6 Fitzpatrick skin types
# uv is a positive number
# complexion is an int between 1 to 6
def reapply_interval(uv, complexion):
    base_time = 0
    if (uv <= 5):
        base_time = 120
    elif (uv <= 8):
        base_time = 90
    else:
        base_time = 70
    
    if (complexion <= 2):
        return base_time - 20
    elif (complexion <= 4):
        return base_time - 10
    else:
        return base_time
    
########################################################################
############### Generating a recommendation for the user ###############
########################################################################

# setting up the connection to Cohere API
co = cohere.Client(config.api_key_cohere)

# Interacting with Cohere /chat endpoint to generate a response based on user's needs
def recommend_sunscreen(complexion, skin_type, location):
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

# Structuring output as a JSON for better transmission of information
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
    return validated_response

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
# print(recommend_sunscreen("Type 3", "dry", "Toronto"))
# print(recommend_sunscreen("Type 2", "normal", "Toronto"))
# recommend_sunscreen("Type 4", "acne-prone", "Australia")